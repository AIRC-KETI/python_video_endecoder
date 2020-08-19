
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

//#define __VERBOSE

#ifdef __cplusplus
 extern "C" {
#endif

#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/frame.h>
#include <libavutil/avassert.h>
#include <libavutil/avstring.h>
#include <libswscale/swscale.h>

struct VideoCodec {
    AVCodec * mCodec;
    AVCodecContext *mCodecContext;
    AVPacket avpkt;
    AVFrame *frame;
    uint8_t *data[4];
    int dst_line_size[8];
    int buffer_size;
    #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
        AVCodecID codec_id;
        int width;
        int height;
    #else
        CodecID codec_id;
    #endif
    AVPixelFormat pixel_fmt;

    #if LIBSWSCALE_VERSION_INT >= AV_VERSION_INT(1,0,0)
        SwsContext* swsCtx = NULL;
    #endif

    #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
    VideoCodec(AVCodecID cid, AVPixelFormat pfmt){
        avcodec_register_all();

        width = 0;
        height = 0;

        codec_id = cid;
        pixel_fmt = pfmt;

        mCodec = avcodec_find_decoder(codec_id);
        mCodecContext = avcodec_alloc_context3(mCodec);

        mCodecContext->codec_id = codec_id;
        mCodecContext->codec_type = AVMEDIA_TYPE_VIDEO;
        mCodecContext->thread_count = 0;

        mCodecContext->flags |= CODEC_FLAG_LOW_DELAY;

        avcodec_open2(mCodecContext, mCodec, NULL);

        av_init_packet(&avpkt);
        frame = av_frame_alloc();

        for(int i=0;i<4;i++){
            data[i] = NULL;
        }
    }
    #else
    VideoCodec(CodecID cid, AVPixelFormat pfmt){
        avcodec_register_all();

        codec_id = cid;
        pixel_fmt = pfmt;

        mCodec = avcodec_find_decoder(codec_id);
        mCodecContext = avcodec_alloc_context3(mCodec);

        mCodecContext->codec_id = codec_id;
        mCodecContext->codec_type = AVMEDIA_TYPE_VIDEO;
        mCodecContext->thread_count = 0;

        mCodecContext->flags |= CODEC_FLAG_LOW_DELAY;

        avcodec_open2(mCodecContext, mCodec, NULL);

        av_init_packet(&avpkt);
        frame = avcodec_alloc_frame();

        for(int i=0;i<4;i++){
            data[i] = NULL;
        }
    }
    #endif

    ~VideoCodec(){
        avcodec_close(mCodecContext);
        av_free(mCodecContext);
        #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
            av_frame_free(&frame);
        #else
            avcodec_free_frame(&frame);
        #endif

        for(int i=0;i<4;i++){
            if(data[i]!=NULL){
                free(data[i]);
            }
        }
    }
};

void convertColorFormat(VideoCodec *f){
    #if LIBSWSCALE_VERSION_INT < AV_VERSION_INT(1,0,0)
        SwsContext* swsCtx = sws_getContext(f->frame->width, f->frame->height, (AVPixelFormat)f->frame->format,f->frame->width, f->frame->height, f->pixel_fmt, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    #else
        f->swsCtx = sws_getCachedContext(f->swsCtx, f->frame->width, f->frame->height, (AVPixelFormat)f->frame->format, f->frame->width, f->frame->height, f->pixel_fmt, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    #endif

    #if LIBSWSCALE_VERSION_INT < AV_VERSION_INT(1,0,0)
        if(f->data[0]==NULL){
            f->buffer_size = av_image_alloc(f->data, f->dst_line_size, f->frame->width, f->frame->height, f->pixel_fmt, 32);
        }
    #else
        if(f->width!=f->frame->width||f->height!=f->frame->height){
            f->width = f->frame->width;
            f->height = f->frame->height;
            for(int i=0;i<4;i++){
                if(f->data[i]!=NULL){
                    free(f->data[i]);
                }
            }
        }
        if(f->data[0]==NULL){
            f->buffer_size = av_image_alloc(f->data, f->dst_line_size, f->frame->width, f->frame->height, f->pixel_fmt, 32);
        }
    #endif

    #if LIBSWSCALE_VERSION_INT < AV_VERSION_INT(1,0,0)
        sws_scale(swsCtx, f->frame->data, f->frame->linesize, 0, f->frame->height, f->data, f->dst_line_size);
    #else
        sws_scale(f->swsCtx, f->frame->data, f->frame->linesize, 0, f->frame->height, f->data, f->dst_line_size);
    #endif

    #if LIBSWSCALE_VERSION_INT < AV_VERSION_INT(1,0,0)
        sws_freeContext(swsCtx);
    #endif
}

void destroy_video_codec(PyObject* codec) {
    delete (VideoCodec*)PyCapsule_GetPointer(codec, "ptr");
}

static PyObject* create_video_codec(PyObject *self, PyObject *args){
    import_array();
    char *codec;
    char *rgb_type;
    if(!PyArg_ParseTuple(args, "ss", &codec, &rgb_type)){
        Py_RETURN_NONE;
    }

    #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
    AVCodecID codec_id;
    #else
    CodecID codec_id;
    #endif
    AVPixelFormat pixel_fmt;

    if(!strcmp(codec, "h265")){
        #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
            codec_id = AV_CODEC_ID_HEVC;
        #else
            #ifdef __VERBOSE
            printf("You can not use H265(HEVC) decoder because your libavcodec version is lower than 55.28.1.\n");
            #endif
            Py_RETURN_NONE;
        #endif
    }
    else if(!strcmp(codec, "h264")){
        #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
            codec_id = AV_CODEC_ID_H264;
        #else
            codec_id = CODEC_ID_H264;
        #endif
    }
    else{
        #ifdef __VERBOSE
        printf("This videocodec doesn't support %s.\nThis videocodec support h264(AVC) and h265(HEVC)", codec);
        #endif
        Py_RETURN_NONE;
    }

    if(!strcmp(rgb_type, "rgb24")){
        pixel_fmt = AV_PIX_FMT_RGB24;
    }
    else if(!strcmp(rgb_type, "bgr24")){
        pixel_fmt = AV_PIX_FMT_BGR24;
    }
    else {
        #ifdef __VERBOSE
        printf("This videocodec doesn't support %s.\nThis videocodec support rgb24 and bgr24", codec);
        #endif
        Py_RETURN_NONE;
    }

    return PyCapsule_New((void*)new VideoCodec(codec_id, pixel_fmt), "ptr", destroy_video_codec);
}

static PyObject* push_frame_data(PyObject *self, PyObject *args) {
    PyObject* pf = NULL;
    char *pkt = NULL;
    int size = -1;

    if(!PyArg_ParseTuple(args, "Os#:push", &pf, &pkt, &size)){
        Py_RETURN_NONE;
    }
    VideoCodec* f = (VideoCodec*)PyCapsule_GetPointer(pf, "ptr");

    #ifdef __VERBOSE
    printf("packet size: %d, content: %s\n", size, pkt);
    #endif

    uint8_t *buffer = (uint8_t*)malloc(size+AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy( buffer, pkt, size);


    f->avpkt.data = buffer;
    f->avpkt.size = size;

    int ret, got;

    while(f->avpkt.size>0){

        ret = avcodec_decode_video2(f->mCodecContext, f->frame, &got, &f->avpkt);

        #ifdef __VERBOSE
        printf("return value. ret: %d\n",ret);
        #endif

        if(ret<0){
            #ifdef __VERBOSE
            printf("Error while decoding\n");
            #endif
            Py_RETURN_NONE;
        }
        if(got){
            if(f->frame->format==AV_PIX_FMT_YUV420P){
                convertColorFormat(f);
                #ifdef __VERBOSE
                printf("decode frame. pix_fmt: YUV420P width: %d, height: %d\n", f->frame->width, f->frame->height);
                printf("buffer size: %d, size 0: %d, size 1: %d, size 2: %d\n", f->buffer_size, sizeof(f->data[0]), sizeof(f->data[1]), sizeof(f->data[2]));
                #endif
                
                npy_intp dims[3] = {f->frame->height, f->frame->width, 3};
                PyObject *ndArray = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, (void *)f->data[0]);

                #ifdef __VERBOSE
                printf("line. size 0: %d, size 1: %d, size 2: %d\n", f->dst_line_size[0], f->dst_line_size[1], f->dst_line_size[2]);
                #endif
                free(buffer);
                return ndArray;
            }
            else{
                convertColorFormat(f);

                #ifdef __VERBOSE
                printf("decode frame. pix_fmt: YUV420P width: %d, height: %d\n", f->frame->width, f->frame->height);
                printf("buffer size: %d, size 0: %d, size 1: %d, size 2: %d\n", f->buffer_size, sizeof(f->data[0]), sizeof(f->data[1]), sizeof(f->data[2]));
                #endif
                
                npy_intp dims[3] = {f->frame->height, f->frame->width, 3};
                PyObject *ndArray = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, (void *)f->data[0]);

                #ifdef __VERBOSE
                printf("line. size 0: %d, size 1: %d, size 2: %d\n", f->dst_line_size[0], f->dst_line_size[1], f->dst_line_size[2]);
                #endif
                free(buffer);
                return ndArray;
            }
        }

        f->avpkt.data += ret;
        f->avpkt.size -= ret;

    }
    av_frame_unref(f->frame);

    Py_RETURN_NONE;
}

static PyMethodDef video_decoder_methods[] = {
        {"create_codec", create_video_codec, METH_VARARGS, "Create video codec. input argument (codec name, output pixel format). codec: h264 and h265 are supported. output pixel format: rgb24 and bgr24 are supported"},
        {"push_frame_data", push_frame_data, METH_VARARGS, "Push frame. and get decoded frame if available."},
        {NULL, NULL, 0, NULL} /* Sentinel */
    };

static struct PyModuleDef video_decoder_definition = { 
    PyModuleDef_HEAD_INIT,
    "videodecoder",
    "A Python module for ffmpeg video codec.",
    -1, 
    video_decoder_methods
};

PyMODINIT_FUNC PyInit_videodecoder(void) {
    Py_Initialize();
    return PyModule_Create(&video_decoder_definition);
}

#ifdef __cplusplus
 }
#endif
