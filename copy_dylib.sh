#!/bin/bash

PROJECT_DIR=$(dirname "`perl -e 'use Cwd "abs_path";print abs_path(shift)' $0`")
mkdir -p "$PROJECT_DIR/NLPGraphTests/Debug"
DYLIB_LIB_COUNT=`find "$PROJECT_DIR/install/3rdparty/boost_1_58_0/stage/lib" -name lib*.dylib | wc -l`
if [ "$DYLIB_LIB_COUNT" != "0" ]; then
	mkdir -p "$PROJECT_DIR/NLPGraphTest/Debug"
	find "$PROJECT_DIR/install/3rdparty/boost_1_58_0/stage/lib" -name lib*.dylib | while read file; do
		cp -v "$file" "$PROJECT_DIR/NLPGraphTests/Debug"
	done
fi
