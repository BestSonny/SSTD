all:
	python setup.py build_ext --inplace
	python ./setup.py install --record install.txt
	rm -rf build
clean:
	cat install.txt | xargs rm -rf

