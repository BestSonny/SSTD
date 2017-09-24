#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <float.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <vector>
#include <math.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.
namespace caffe {

using namespace boost::property_tree;  // NOLINT(build/namespaces)
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

// Parse VOC/ILSVRC detection annotation.
template <typename T>
bool ReadXMLToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    T anno_datum) {
  ptree pt;
  read_xml(labelfile, pt);

  // Parse annotation.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
    ptree pt1 = v1.second;
    if (v1.first == "object") {
      Annotation* anno = NULL;
      bool difficult = false;
      ptree object = v1.second;
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
        ptree pt2 = v2.second;
        if (v2.first == "name") {
          string name = pt2.data();
          if (name_to_label.find(name) == name_to_label.end()) {
            LOG(FATAL) << "Unknown name: " << name;
          }
          int label = name_to_label.find(name)->second;
          bool found_group = false;
          for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
            AnnotationGroup* anno_group =
                anno_datum->mutable_annotation_group(g);
            if (label == anno_group->group_label()) {
              if (anno_group->annotation_size() == 0) {
                instance_id = 0;
              } else {
                instance_id = anno_group->annotation(
                    anno_group->annotation_size() - 1).instance_id() + 1;
              }
              anno = anno_group->add_annotation();
              found_group = true;
            }
          }
          if (!found_group) {
            // If there is no such annotation_group, create a new one.
            AnnotationGroup* anno_group = anno_datum->add_annotation_group();
            anno_group->set_group_label(label);
            anno = anno_group->add_annotation();
            instance_id = 0;
          }
          anno->set_instance_id(instance_id++);
        } else if (v2.first == "difficult") {
          difficult = pt2.data() == "1";
        } else if (v2.first == "bndbox") {
          int xmin = pt2.get("xmin", 0);
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);
          int ymax = pt2.get("ymax", 0);
          CHECK_NOTNULL(anno);
          LOG_IF(WARNING, xmin > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax > width) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax > height) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymin < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, ymax < 0) << labelfile <<
              " bounding box exceeds image boundary.";
          LOG_IF(WARNING, xmin > xmax) << labelfile <<
              " bounding box irregular.";
          LOG_IF(WARNING, ymin > ymax) << labelfile <<
              " bounding box irregular.";
          // Store the normalized bounding box.
          NormalizedBBox* bbox = anno->mutable_bbox();
          bbox->set_xmin(static_cast<float>(xmin) / width);
          bbox->set_ymin(static_cast<float>(ymin) / height);
          bbox->set_xmax(static_cast<float>(xmax) / width);
          bbox->set_ymax(static_cast<float>(ymax) / height);
          bbox->set_difficult(difficult);
        }
      }
    }
  }
  return true;
}

// Parse MSCOCO detection annotation.
template <typename T>
bool ReadJSONToAnnotatedDatum(const string& labelfile, const int img_height,
    const int img_width, const std::map<string, int>& name_to_label,
    T anno_datum) {
  ptree pt;
  read_json(labelfile, pt);

  // Get image info.
  int width = 0, height = 0;
  try {
    height = pt.get<int>("image.height");
    width = pt.get<int>("image.width");
  } catch (const ptree_error &e) {
    LOG(WARNING) << "When parsing " << labelfile << ": " << e.what();
    height = img_height;
    width = img_width;
  }
  LOG_IF(WARNING, height != img_height) << labelfile <<
      " inconsistent image height.";
  LOG_IF(WARNING, width != img_width) << labelfile <<
      " inconsistent image width.";
  CHECK(width != 0 && height != 0) << labelfile <<
      " no valid image width/height.";

  // Get annotation info.
  int instance_id = 0;
  BOOST_FOREACH(ptree::value_type& v1, pt.get_child("annotation")) {
    Annotation* anno = NULL;
    bool iscrowd = false;
    ptree object = v1.second;
    // Get category_id.
    string name = object.get<string>("category_id");
    if (name_to_label.find(name) == name_to_label.end()) {
      LOG(FATAL) << "Unknown name: " << name;
    }
    int label = name_to_label.find(name)->second;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group =
          anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);

    // Get iscrowd.
    iscrowd = object.get<int>("iscrowd", 0);

    // Get bbox.
    vector<float> bbox_items;
    BOOST_FOREACH(ptree::value_type& v2, object.get_child("bbox")) {
      bbox_items.push_back(v2.second.get_value<float>());
    }
    CHECK_EQ(bbox_items.size(), 4);
    float xmin = bbox_items[0];
    float ymin = bbox_items[1];
    float xmax = bbox_items[0] + bbox_items[2];
    float ymax = bbox_items[1] + bbox_items[3];
    CHECK_NOTNULL(anno);
    LOG_IF(WARNING, xmin > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
        " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
        " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
        " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(iscrowd);

    vector<float> oriented_bbox_items;
    BOOST_FOREACH(ptree::value_type& v2, object.get_child("segmentation")) {
      ptree list_seg = v2.second;
      BOOST_FOREACH(ptree::value_type& v3,list_seg){
        oriented_bbox_items.push_back(v3.second.get_value<float>());
      }
    }
    CHECK_EQ(oriented_bbox_items.size(), 8);
    float xl_t = oriented_bbox_items[0];
    float yl_t = oriented_bbox_items[1];
    float xr_t = oriented_bbox_items[2];
    float yr_t = oriented_bbox_items[3];
    float xr_b = oriented_bbox_items[4];
    float yr_b = oriented_bbox_items[5];
    float xl_b = oriented_bbox_items[6];
    float yl_b = oriented_bbox_items[7];

    cv::Point2f point1 = cv::Point2f(xl_t,yl_t);
    cv::Point2f point2 = cv::Point2f(xr_t,yr_t);
    cv::Point2f point3 = cv::Point2f(xr_b,yr_b);
    cv::Point2f point4 = cv::Point2f(xl_b,yl_b);

    float radians;
    cv::Point2f center = 0.5f * (point1 + point3);
    cv::Size2f size;
    cv::Vec2f vecs[2];
    vecs[0] = cv::Vec2f(point1 - point2);
    vecs[1] = cv::Vec2f(point2 - point3);
    const double PI = 3.14159265;
    // check that given sides are not perpendicular
    if (fabs(vecs[0].dot(vecs[1])) / (cv::norm(vecs[0])* cv::norm(vecs[1])) > FLT_EPSILON){
      radians = atan(vecs[0][1] / vecs[0][0]);
      float _width = (float) cv::norm(vecs[0]);
      float _height = (float) cv::norm(vecs[1]);
      if (fabs(vecs[0][1] / vecs[0][0]) > 1){
        radians = atan(vecs[1][1] / vecs[1][0]);
        _width = (float) cv::norm(vecs[1]);
        _height = (float) cv::norm(vecs[0]);
      }
      size = cv::Size2f(_width, _height);
    }else{
      float _width = (float) cv::norm(vecs[0]);
      float _height = (float) cv::norm(vecs[1]);
      if (fabs(vecs[0][1] / (1e-6 +vecs[0][0])) > 1){
        _width = (float) cv::norm(vecs[1]);
        _height = (float) cv::norm(vecs[0]);
      }
      size = cv::Size2f(_width, _height);
      radians = 0;
    }
    float xc = center.x;
    float yc = center.y;
    float x_width = size.width;
    float x_height = size.height;
    // float xc = 0.25*(xl_t + xr_t + xr_b + xl_b);
    // float yc = 0.25*(yl_t + yr_t + yr_b + yl_b);
    // float dy1 = (fabs(yl_t - yr_t) > 2 ? (yr_t - yl_t) :1e-10);
    // float dx1 = (fabs(xl_t - xr_t) > 2 ? (xl_t - xr_t) :1e-10);
    // float dy2 = (fabs(yr_b - yl_b) > 2 ? (yr_b - yl_b) :1e-10);
    // float dx2 = (fabs(xl_b - xr_b) > 2 ? (xl_b - xr_b) :1e-10);
    // float radians = atan(0.5 * dy1 / dx1 + 0.5* dy2 / dx2);
    // float x0 = xl_t - xc;
    // float x1 = xr_t - xc;
    // float x2 = xr_b - xc;
    // float x3 = xl_b - xc;
    // float y0 = yl_t - yc;
    // float y1 = yr_t - yc;
    // float y2 = yr_b - yc;
    // float y3 = yl_b - yc;

    // xl_t = cos(radians) * x0 - sin(radians)*y0 + xc;
    // xr_t = cos(radians) * x1 - sin(radians)*y1 + xc;
    // xr_b = cos(radians) * x2 - sin(radians)*y2 + xc;
    // xl_b = cos(radians) * x3 - sin(radians)*y3 + xc;
    // yl_t = sin(radians) * x0 + cos(radians)*y0 + yc;
    // yr_t = sin(radians) * x1 + cos(radians)*y1 + yc;
    // yr_b = sin(radians) * x2 + cos(radians)*y2 + yc;
    // yl_b = sin(radians) * x3 + cos(radians)*y3 + yc;

    // LOG(INFO) << xl_t << " "
    //      << yl_t << " "
    //      << xr_t << " "
    //      << yr_t << " "
    //      << xr_b << " "
    //      << yr_b << " "
    //      << xl_b << " "
    //      << yl_b << " "
    //      << std::endl;
    // float x_width = 0.5* (std::abs(xr_t - xl_t)+std::abs(xr_b - xl_b));
    // float x_height =0.5* (std::abs(yr_b - yr_t)+std::abs(yr_b - yl_b));

    // CHECK_LT(std::abs(xc-xl_t-0.5*x_width) , 5);
    // CHECK_LT(std::abs(yc-yl_t-0.5*x_height), 5);
    // CHECK_GT(x_width, 0);
    // CHECK_GT(x_height, 0);

    CHECK_NOTNULL(anno);
    // Store the normalized oriented bounding box.
    NormalizedOrientedBBox* oriented_bbox = anno->mutable_oriented_bbox();
    oriented_bbox->set_xc(xc / (1.0 * width));
    oriented_bbox->set_yc(yc / (1.0 * height));
    oriented_bbox->set_width(x_width / (1.0 * width));
    oriented_bbox->set_height(x_height / (1.0 * height));
    oriented_bbox->set_radians(radians);
    // LOG(INFO) << oriented_bbox->xc() << " "
    //      << oriented_bbox->yc() << " "
    //      << oriented_bbox->width() << " "
    //      << oriented_bbox->height() << " "
    //      << oriented_bbox->radians() << ""
    //      << std::endl;
  }
  return true;
}

// Parse plain txt detection annotation: label_id, xmin, ymin, xmax, ymax.
template <typename T>
bool ReadTxtToAnnotatedDatum(const string& labelfile, const int height,
    const int width, T anno_datum) {
  std::ifstream infile(labelfile.c_str());
  if (!infile.good()) {
    LOG(INFO) << "Cannot open " << labelfile;
    return false;
  }
  int label;
  float xmin, ymin, xmax, ymax;
  while (infile >> label >> xmin >> ymin >> xmax >> ymax) {
    Annotation* anno = NULL;
    int instance_id = 0;
    bool found_group = false;
    for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
      AnnotationGroup* anno_group = anno_datum->mutable_annotation_group(g);
      if (label == anno_group->group_label()) {
        if (anno_group->annotation_size() == 0) {
          instance_id = 0;
        } else {
          instance_id = anno_group->annotation(
              anno_group->annotation_size() - 1).instance_id() + 1;
        }
        anno = anno_group->add_annotation();
        found_group = true;
      }
    }
    if (!found_group) {
      // If there is no such annotation_group, create a new one.
      AnnotationGroup* anno_group = anno_datum->add_annotation_group();
      anno_group->set_group_label(label);
      anno = anno_group->add_annotation();
      instance_id = 0;
    }
    anno->set_instance_id(instance_id++);
    LOG_IF(WARNING, xmin > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax > width) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax > height) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymin < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, ymax < 0) << labelfile <<
      " bounding box exceeds image boundary.";
    LOG_IF(WARNING, xmin > xmax) << labelfile <<
      " bounding box irregular.";
    LOG_IF(WARNING, ymin > ymax) << labelfile <<
      " bounding box irregular.";
    // Store the normalized bounding box.
    NormalizedBBox* bbox = anno->mutable_bbox();
    bbox->set_xmin(xmin / width);
    bbox->set_ymin(ymin / height);
    bbox->set_xmax(xmax / width);
    bbox->set_ymax(ymax / height);
    bbox->set_difficult(false);
  }
  return true;
}

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim,
    const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (min_dim > 0 || max_dim > 0) {
    int num_rows = cv_img_origin.rows;
    int num_cols = cv_img_origin.cols;
    int min_num = std::min(num_rows, num_cols);
    int max_num = std::max(num_rows, num_cols);
    float scale_factor = 1;
    if (min_dim > 0 && min_num < min_dim) {
      scale_factor = static_cast<float>(min_dim) / min_num;
    }
    if (max_dim > 0 && static_cast<int>(scale_factor * max_num) > max_dim) {
      // Make sure the maximum dimension is less than max_dim.
      scale_factor = static_cast<float>(max_dim) / max_num;
    }
    if (scale_factor == 1) {
      cv_img = cv_img_origin;
    } else {
      cv::resize(cv_img_origin, cv_img, cv::Size(0, 0),
                 scale_factor, scale_factor);
    }
  } else if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename, const int height,
    const int width, const int min_dim, const int max_dim) {
  return ReadImageToCVMat(filename, height, width, min_dim, max_dim, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  return ReadImageToCVMat(filename, height, width, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.') + 1;
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const int min_dim, const int max_dim,
    const bool is_color, const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim,
                                    is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          !min_dim && !max_dim && matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      EncodeCVMatToDatum(cv_img, encoding, datum);
      datum->set_label(label);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

void GetImageSize(const string& filename, int* height, int* width) {
  cv::Mat cv_img = cv::imread(filename);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return;
  }
  *height = cv_img.rows;
  *width = cv_img.cols;
}

bool ReadRichImageToAnnotatedDatum(const string& filename,
    const string& labelfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const string& encoding, const AnnotatedDatum_AnnotationType type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedDatum* anno_datum) {
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  anno_datum->clear_annotation_group();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  switch (type) {
    case AnnotatedDatum_AnnotationType_BBOX:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "xml") {
        return ReadXMLToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       name_to_label, anno_datum);
      } else if (labeltype == "json") {
        return ReadJSONToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        name_to_label, anno_datum);
      } else if (labeltype == "txt") {
        return ReadTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

bool ReadRichImageToAnnotatedMaskDatum(const string& filename,
    const string& labelfile, const string& maskfile, const int height, const int width,
    const int min_dim, const int max_dim, const bool is_color,
    const std::string& encoding, const AnnotatedMaskDatum_AnnotationType type,
    const string& labeltype, const std::map<string, int>& name_to_label,
    AnnotatedMaskDatum* anno_datum) {
  // Read image to datum.
  bool status = ReadImageToDatum(filename, -1, height, width,
                                 min_dim, max_dim, is_color, encoding,
                                 anno_datum->mutable_datum());
  if (status == false) {
    return status;
  }
  status = ReadImageToDatum(maskfile, -1, height, width,
                                 min_dim, max_dim, false, "png",
                                 anno_datum->mutable_mask());
  if (status == false) {
    return status;
  }
  anno_datum->clear_annotation_group();
  if (!boost::filesystem::exists(labelfile)) {
    return true;
  }
  switch (type) {
    case AnnotatedMaskDatum_AnnotationType_BBOX:
      int ori_height, ori_width;
      GetImageSize(filename, &ori_height, &ori_width);
      if (labeltype == "xml") {
        return ReadXMLToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       name_to_label, anno_datum);
      } else if (labeltype == "json") {
        return ReadJSONToAnnotatedDatum(labelfile, ori_height, ori_width,
                                        name_to_label, anno_datum);
      } else if (labeltype == "txt") {
        return ReadTxtToAnnotatedDatum(labelfile, ori_height, ori_width,
                                       anno_datum);
      } else {
        LOG(FATAL) << "Unknown label file type.";
        return false;
      }
      break;
    default:
      LOG(FATAL) << "Unknown annotation type.";
      return false;
  }
}

#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

bool ReadLabelFileToLabelMap(const string& filename, bool include_background,
    const string& delimiter, LabelMap* map) {
  // cleanup
  map->Clear();

  std::ifstream file(filename.c_str());
  string line;
  // Every line can have [1, 3] number of fields.
  // The delimiter between fields can be one of " :;".
  // The order of the fields are:
  //  name [label] [display_name]
  //  ...
  int field_size = -1;
  int label = 0;
  LabelMapItem* map_item;
  // Add background (none_of_the_above) class.
  if (include_background) {
    map_item = map->add_item();
    map_item->set_name("none_of_the_above");
    map_item->set_label(label++);
    map_item->set_display_name("background");
  }
  while (std::getline(file, line)) {
    vector<string> fields;
    fields.clear();
    boost::split(fields, line, boost::is_any_of(delimiter));
    if (field_size == -1) {
      field_size = fields.size();
    } else {
      CHECK_EQ(field_size, fields.size())
          << "Inconsistent number of fields per line.";
    }
    map_item = map->add_item();
    map_item->set_name(fields[0]);
    switch (field_size) {
      case 1:
        map_item->set_label(label++);
        map_item->set_display_name(fields[0]);
        break;
      case 2:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[0]);
        break;
      case 3:
        label = std::atoi(fields[1].c_str());
        map_item->set_label(label);
        map_item->set_display_name(fields[2]);
        break;
      default:
        LOG(FATAL) << "The number of fields should be [1, 3].";
        break;
    }
  }
  return true;
}

bool MapNameToLabel(const LabelMap& map, const bool strict_check,
    std::map<string, int>* name_to_label) {
  // cleanup
  name_to_label->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!name_to_label->insert(std::make_pair(name, label)).second) {
        LOG(FATAL) << "There are many duplicates of name: " << name;
        return false;
      }
    } else {
      (*name_to_label)[name] = label;
    }
  }
  return true;
}

bool MapLabelToName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_name) {
  // cleanup
  label_to_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_name->insert(std::make_pair(label, name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_name)[label] = name;
    }
  }
  return true;
}

bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_display_name) {
  // cleanup
  label_to_display_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& display_name = map.item(i).display_name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_display_name->insert(
              std::make_pair(label, display_name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_display_name)[label] = display_name;
    }
  }
  return true;
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum) {
  std::vector<uchar> buf;
  cv::imencode("."+encoding, cv_img, buf);
  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                              buf.size()));
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_encoded(true);
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
