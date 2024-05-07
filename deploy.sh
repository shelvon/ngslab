#!/usr/bin/bash

name="ngslab"

build_type="pip"

while [ $# -gt 0 ]; do
	opt="$1"
	case $opt in
	-build_type)
	  shift
	  build_type="$1"
	  echo "$0 -build_type ${build_type}"
	  shift
	  ;;
	*)
	  shift
	  ;;
	esac

done

prj_path=$(pwd)

# step 1
python3 -m build

# step 2
if [[ ${build_type} == "pip" ]]; then

	# remove the package
	pip uninstall ${name}
	# local installation
	python3 -m pip install --upgrade --no-index --find-links="file://${prj_path}/dist" ${name}

# installation from conda is much slower
elif [[ ${build_type} == "conda" ]]; then

	conda uninstall ${name}
	# conda build purge #  remove local build intermediates
	conda build 'conda_build/'
	conda install --use-local ${name}

fi
