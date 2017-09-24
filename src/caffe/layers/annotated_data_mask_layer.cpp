#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>
#include <iostream>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_mask_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"



namespace caffe {

template <typename Dtype>
AnnotatedDataMaskLayer<Dtype>::AnnotatedDataMaskLayer(const LayerParameter& param)
  : BaseDataMaskLayer<Dtype>(param),
    prefetch_free_(), prefetch_full_(),
    reader_(param){
      for (int i = 0; i < PREFETCH_COUNT; ++i) {
        prefetch_free_.push(&prefetch_[i]);
      }
    DLOG(INFO) << "Construtor completed";
}

template <typename Dtype>
AnnotatedDataMaskLayer<Dtype>::~AnnotatedDataMaskLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataMaskLayer<Dtype>::DataLayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  oriented_ = anno_data_param.oriented();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();

  // Read a data point, and use it to initialize the top blob.
  AnnotatedMaskDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> data_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(data_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  data_shape[0] = batch_size;
  top[0]->Reshape(data_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(data_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> mask_shape =
        this->data_transformer_->InferBlobShape(anno_datum.mask());
    this->transformed_mask_.Reshape(mask_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    mask_shape[0] = batch_size;
    top[1]->Reshape(mask_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].mask_.Reshape(mask_shape);
    }
    LOG(INFO) << "output mask size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
    has_anno_type_ = anno_datum.has_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedMaskDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
        if(oriented_){
          label_shape[3] = 13;
        }
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[2]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}
template <typename Dtype>
void AnnotatedDataMaskLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataMaskLayer<Dtype>::LayerSetUp(bottom, top);
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].mask_.mutable_cpu_data();
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].mask_.mutable_gpu_data();
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  this->mask_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void AnnotatedDataMaskLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif
  try{
    while (!must_stop()) {
      BatchMask<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
  // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void AnnotatedDataMaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BatchMask<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->mask_);
    top[2]->ReshapeLike(batch->label_);
    // Copy the masks and labels.
    caffe_copy(batch->mask_.count(), batch->mask_.cpu_data(),
        top[1]->mutable_cpu_data());
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[2]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataMaskLayer<Dtype>::load_batch(BatchMask<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  CHECK(this->transformed_mask_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  AnnotatedMaskDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> data_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(data_shape);
  // Reshape batch according to the batch_size.
  data_shape[0] = batch_size;
  batch->data_.Reshape(data_shape);
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* mask_data = NULL;
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    vector<int> mask_shape =
        this->mask_transformer_->InferBlobShape(anno_datum.mask());
    this->transformed_mask_.Reshape(mask_shape);
    mask_shape[0] = batch_size;
    batch->mask_.Reshape(mask_shape);
    mask_data = batch->mask_.mutable_cpu_data();
  }
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    AnnotatedMaskDatum& temp_anno_datum = *(reader_.full().pop("Waiting for data"));
    AnnotatedMaskDatum anno_datum(temp_anno_datum);
    this->data_transformer_->DistortImage(temp_anno_datum.datum(),
                                          anno_datum.mutable_datum());
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedMaskDatum sampled_datum;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from anno_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(anno_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the anno_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        this->data_transformer_->CropImage(anno_datum, sampled_bboxes[rand_idx],
                                           &sampled_datum, false);
        this->mask_transformer_->CropImage(anno_datum, sampled_bboxes[rand_idx],
                                           &sampled_datum, true);
      } else {
        sampled_datum.CopyFrom(anno_datum);
      }
    } else {
      sampled_datum.CopyFrom(anno_datum);
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      int offset_mask = batch->mask_.offset(item_id);
      this->transformed_mask_.set_cpu_data(mask_data + offset_mask);
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum.has_type()) << "Some datum misses AnnotationType.";
        CHECK_EQ(anno_type_, sampled_datum.type()) <<
            "Different AnnotationType.";
        // Keep same mirror
        this->data_transformer_->setForceMirrorFlag(true);
        this->mask_transformer_->setForceMirrorFlag(true);
        bool do_mirror = this->data_transformer_->getMirror();
        this->data_transformer_->setMirror(do_mirror);
        this->mask_transformer_->setMirror(do_mirror);
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec, false);
        transformed_anno_vec.clear();
        this->mask_transformer_->Transform(sampled_datum,
                                           &(this->transformed_mask_),
                                           &transformed_anno_vec, true);
        // vector<int> shape2 =
        //    this->data_transformer_->InferBlobShape(sampled_datum.mask());
        //std::cout << "mask " << shape2[0] << " " << shape2[1] << " " << shape2[2] << " " << shape2[3] << std::endl;
        if (anno_type_ == AnnotatedMaskDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum.datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum.datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum.datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum.datum(),
                                         &(this->transformed_data_));
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedMaskDatum*>(&temp_anno_datum));
  }
  // Store "rich" annotation if needed.
  int shape_annotation = 8;
  if(oriented_){
    shape_annotation = 13;
  }
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedMaskDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = shape_annotation;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(shape_annotation, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
              if(shape_annotation == 13){
                const NormalizedOrientedBBox& oriented_bbox = anno.oriented_bbox();
                top_label[idx++] = oriented_bbox.xc();
                top_label[idx++] = oriented_bbox.yc();
                top_label[idx++] = oriented_bbox.width();
                top_label[idx++] = oriented_bbox.height();
                top_label[idx++] = oriented_bbox.radians();
                // std::cout << oriented_bbox.xc() << " "
                //      << oriented_bbox.yc() << " "
                //      << oriented_bbox.width() << " "
                //      << oriented_bbox.height() << " "
                //      << oriented_bbox.radians() << ""
                //      << std::endl;
              }
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(AnnotatedDataMaskLayer, Forward);
#endif

INSTANTIATE_CLASS(AnnotatedDataMaskLayer);
REGISTER_LAYER_CLASS(AnnotatedDataMask);

}  // namespace caffe
