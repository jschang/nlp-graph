#!/bin/bash

OS=`uname`
if [ $OS = 'Linux' ]; then
	sudo apt-get update
	sudo apt-get install postgresql python-dev build-essential g++ libbz2-dev
fi;
if [ $OS = 'Darwin' ]; then
POSTGRES_HOME=/Applications/Postgres.app/Contents/Versions/9.4
else
POSTGRES_HOME=/usr/lib/postgresql/9.1
fi;
PROJECT_DIR=$(dirname "`perl -e 'use Cwd "abs_path";print abs_path(shift)' $0`")
BUILD_DIR="$PROJECT_DIR/build/3rdparty"
INSTALL_DIR="$PROJECT_DIR/install/3rdparty"
ARCH_DIR="$PROJECT_DIR/3rdparty"
PATH=$PATH:$POSTGRES_HOME/bin

chkerr() {
    if [ "$?" != "0" ]; then
        echo "$1"
        exit 1;
    fi
}

untar() {
    if [ -d "$PROJECT_DIR/$1/3rdparty/$2" ]; then
        echo "$2 already untarred" 
        return 0
    fi
    tar xzvf "$ARCH_DIR/$2.tar.gz" -C "$PROJECT_DIR/$1/3rdparty"
    chkerr "Unable to untar $2"
}

do_make() {
    _TARGET="$1"
    shift
    PREFIX="$INSTALL_DIR/$_TARGET"
    if [ -d "$PREFIX" ]; then
        echo "Install already exists at $PREFIX"
        return 0
    fi
    untar build $_TARGET
    pushd "$BUILD_DIR/$_TARGET"
    make $@
    popd
}

do_cmake() {
    _TARGET="$1"
    shift
    PREFIX="$INSTALL_DIR/$_TARGET"
    if [ -d "$PREFIX" ]; then
        echo "Install already exists at $PREFIX"
        return 0
    fi
    untar build $_TARGET
    pushd "$BUILD_DIR/$_TARGET"
    cmake $@
    popd
}

do_configure() {
    _TARGET="$1"
    shift
    PREFIX="$INSTALL_DIR/$_TARGET"
    if [ -d "$PREFIX" ]; then
        echo "Install already exists at $PREFIX"
        return 0
    fi
    untar build $_TARGET
    pushd "$BUILD_DIR/$_TARGET"
    ./configure --prefix="$PREFIX" $@
    chkerr "Unable to configure $_TARGET"
    popd
}

make_install_boost() {
    _TARGET="$1"
    shift
    PREFIX="$INSTALL_DIR/$_TARGET"
    pushd "$INSTALL_DIR/$_TARGET"
    ./bootstrap.sh
    chkerr "Unable to bootstrap $_TARGET"
    ./b2 -a link=static runtime-link=static variant=release threading=multi
    chkerr "Unable to build $_TARGET"
    ./b2 --with-log link=shared runtime-link=shared variant=release threading=multi
    chkerr "Unable to build $_TARGET library log"
    #_FILE="$PREFIX/stage/lib/libboost_unit_test_framework.dylib" 
    #rm "$_FILE"
    #chkerr "Unable to remove $_FILE"
    popd
}

install() {
    
    mkdir -pv "$PROJECT_DIR" build/3rdparty
    chkerr "Unable to create build directory";
    mkdir -pv "$PROJECT_DIR" install/3rdparty
    chkerr "Unable to create install directory";
    
    untar install "boost_1_58_0"
    make_install_boost "boost_1_58_0"
    
    untar install "compute-0.4"
    
    do_configure "libpqxx-4.0.1"
    do_make "libpqxx-4.0.1"
    chkerr "Unable to make $_TARGET"
    do_make "libpqxx-4.0.1" install
    # this will fail, but only at the archive install...
    # so we'll just copy that manually
    cp "$BUILD_DIR/libpqxx-4.0.1/src/.libs/libpqxx.a" "$INSTALL_DIR/$_TARGET/lib"
    
    do_cmake "gtest-1.7.0"
    do_make "gtest-1.7.0"
}

clean() {
    rm -rf "$PROJECT_DIR/build/3rdparty"
    rm -rf "$PROJECT_DIR/install/3rdparty"
}

case $1 in 
install|clean)
    eval $1
    ;;
*)
    echo clean or install are the only options
    ;;
esac



