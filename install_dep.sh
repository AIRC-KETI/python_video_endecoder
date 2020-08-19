apt update
apt-get update -qq && apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libsdl2-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  pkg-config \
  texinfo \
  wget \
  zlib1g-dev

mkdir -p ~/ffmpeg_sources ~/bin

apt-get install -y nasm
apt-get install -y yasm
apt-get install -y libx264-dev
apt-get install -y libx265-dev libnuma-dev
apt-get install -y libvpx-dev
apt-get install -y libfdk-aac-dev
apt-get install -y libmp3lame-dev
apt-get install -y libopus-dev

cd ~/ffmpeg_sources && \
wget -O ffmpeg-3.2.tar.bz2 http://ffmpeg.org/releases/ffmpeg-3.2.tar.bz2 && \
tar xjvf ffmpeg-3.2.tar.bz2 && \
cd ffmpeg-3.2 && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm -fPIC" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-shared \
  --enable-pic \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r

cp $HOME/ffmpeg_build/lib/pkgconfig/* /usr/lib/pkgconfig
cp $HOME/ffmpeg_build/lib/lib* /usr/lib/
cp $HOME/bin/* /bin

apt-get install -y python3 python3-pip
pip3 install numpy

#  --pkg-config-flags="--static" \
  # --enable-shared \
  # --enable-pic \
