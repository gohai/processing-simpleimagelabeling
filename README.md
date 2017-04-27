# Simple Image Labeling

see [examples/Test/Test.pde](examples/Test/Test.pde)

## TODO

* build a Pi 2/3 optimized version of the JNI library
* look into using different models (different, more labels?)
* attempt to make the code generic so that it can work with different models
* expose GraphBuilder class?

## Notes for compiling the JNI bindings for TensorFlow on ARM Linux

The repository comes with libtensorflow_jni.so for armv6hf, but if this needs to be re-done with a later version of TensorFlow:

    sudo dd if=/dev/zero of=/dev/swap bs=1M count=1000
    sudo mkswap /swap
    sudo chmod 0600 /swap
    sudo swapon /swap

    sudo apt-get install -y autoconf automake libtool gcc-4.8 g++-4.8 zip
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100

    sudo cp bezel /usr/local/bin

    wget https://github.com/tensorflow/tensorflow/archive/v1.1.0-rc2.tar.gz
    tar vfx v1.1.0-rc2.tar.gz
    cd tensorflow-1.1.0-rc2

    unset IS_MOBILE_PLATFORM in tensorflow/core/platform/platform.h to fix https://github.com/tensorflow/tensorflow/issues/3469

    ./configure
    passed -Os as compile flag, rest default
    alternatively, to optimize for Pi2 and Pi3: -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize -Os
    something else that was suggested somewhere: -D__ANDROID_TYPES_SLIM__
    bazel build --config opt tensorflow/java:tensorflow tensorflow/java:libtensorflow_jni
