
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
#include <libavutil/mathematics.h>
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
    int mWidth;
    int mHeight;
    int mBitrate;
    int mIFrameInterval;
    int mFrameRate;
    int pts;

    int in_linesize[1];
    uint8_t *rgb[1];

    #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
        AVCodecID codec_id;
    #else
        CodecID codec_id;
    #endif
    AVPixelFormat pixel_fmt;

    #if LIBSWSCALE_VERSION_INT >= AV_VERSION_INT(1,0,0)
        SwsContext* swsCtx = NULL;
    #endif

    #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
    VideoCodec(AVCodecID cid, AVPixelFormat pfmt, int w, int h, int bitrate, int iframeinterval, int framerate){
        avcodec_register_all();

        mWidth = w;
        mHeight = h;
        mIFrameInterval = iframeinterval;
        mFrameRate = framerate;

        pts = 0;

        codec_id = cid;
        pixel_fmt = pfmt;

        #ifdef __VERBOSE
        printf("find encoder and alloc context\n");
        #endif

        mCodec = avcodec_find_encoder(codec_id);
        mCodecContext = avcodec_alloc_context3(mCodec);

        mCodecContext->bit_rate = bitrate;
        mCodecContext->width = w;
        mCodecContext->height = h;
        mCodecContext->time_base = (AVRational){1, framerate};
        mCodecContext->gop_size = iframeinterval;
        mCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;

        in_linesize[0] = w*3;

        #ifdef __VERBOSE
        printf("alloc rgb buffer\n");
        #endif

        rgb[0] = (uint8_t *)malloc( 3 * sizeof(uint8_t) * w * h);

        av_opt_set(mCodecContext->priv_data, "preset", "ultrafast", 0);
        av_opt_set(mCodecContext->priv_data, "tune", "zerolatency", 0);

        avcodec_open2(mCodecContext, mCodec, NULL);

        #ifdef __VERBOSE
        printf("av frame alloc\n");
        #endif

        av_init_packet(&avpkt);
        frame = av_frame_alloc();
        frame->format = mCodecContext->pix_fmt;
        frame->width = mCodecContext->width;
        frame->height = mCodecContext->height;

        #ifdef __VERBOSE
        printf("av frame buffer alloc\n");
        #endif

        int ret = av_image_alloc(frame->data, frame->linesize, frame->width, frame->height, (AVPixelFormat)frame->format, 32);
    }
    #else
    VideoCodec(CodecID cid, AVPixelFormat pfmt, int w, int h, int bitrate, int iframeinterval, int framerate){
        avcodec_register_all();

        mWidth = w;
        mHeight = h;
        mIFrameInterval = iframeinterval;
        mFrameRate = framerate;

        pts = 0;

        codec_id = cid;
        pixel_fmt = pfmt;

        #ifdef __VERBOSE
        printf("find encoder and alloc context\n");
        #endif

        mCodec = avcodec_find_encoder(codec_id);
        mCodecContext = avcodec_alloc_context3(mCodec);

        mCodecContext->bit_rate = bitrate;
        mCodecContext->width = w;
        mCodecContext->height = h;
        mCodecContext->time_base = (AVRational){1, framerate};
        mCodecContext->gop_size = iframeinterval;
        mCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;

        in_linesize[0] = w*3;

        #ifdef __VERBOSE
        printf("alloc rgb buffer\n");
        #endif

        //rgb[0] = (uint8_t *)realloc((void *)rgb[0], 3 * sizeof(uint8_t) * w * h);
        rgb[0] = (uint8_t *)malloc( 3 * sizeof(uint8_t) * w * h);

        av_opt_set(mCodecContext->priv_data, "preset", "ultrafast", 0);
        av_opt_set(mCodecContext->priv_data, "tune", "zerolatency", 0);

        avcodec_open2(mCodecContext, mCodec, NULL);

        #ifdef __VERBOSE
        printf("av frame alloc\n");
        #endif

        frame = avcodec_alloc_frame();
        frame->format = mCodecContext->pix_fmt;
        frame->width = mCodecContext->width;
        frame->height = mCodecContext->height;

        #ifdef __VERBOSE
        printf("av frame buffer alloc\n");
        #endif

        int ret = av_image_alloc(frame->data, frame->linesize, frame->width, frame->height, (AVPixelFormat)frame->format, 32);
    }
    #endif

    ~VideoCodec(){
        avcodec_close(mCodecContext);
        av_free(mCodecContext);

        av_freep(&frame->data[0]);

        #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(55,28,1)
            av_frame_free(&frame);
            printf("av_frame_free(&frame);\n");
        #else
            avcodec_free_frame(&frame);
        #endif

        free(rgb[0]);
    }
};

void convertColorFormat(VideoCodec *f){
    #if LIBSWSCALE_VERSION_INT < AV_VERSION_INT(1,0,0)
        SwsContext* swsCtx = sws_getContext(f->frame->width, f->frame->height, (AVPixelFormat)f->pixel_fmt,f->frame->width, f->frame->height, (AVPixelFormat)f->frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    #else
        f->swsCtx = sws_getCachedContext(f->swsCtx, f->frame->width, f->frame->height, (AVPixelFormat)f->pixel_fmt, f->frame->width, f->frame->height, (AVPixelFormat)f->frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    #endif

    #ifdef __VERBOSE
        printf("configured sw scale\n");
    #endif

    #if LIBSWSCALE_VERSION_INT < AV_VERSION_INT(1,0,0)
        sws_scale(swsCtx,f->rgb, f->in_linesize, 0, f->frame->height, f->frame->data , f->frame->linesize);
    #else
        sws_scale(f->swsCtx,f->rgb, f->in_linesize, 0, f->frame->height, f->frame->data , f->frame->linesize);
    #endif

    #ifdef __VERBOSE
        printf("convert complete\n");
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

    int w, h, bitrate, iframeinterval, framerate;

    #ifdef __VERBOSE
        printf("start parsing\n");
    #endif

    if(!PyArg_ParseTuple(args, "ssiiiii", &codec, &rgb_type, &w, &h, &bitrate, &iframeinterval, &framerate)){
        Py_RETURN_NONE;
    }

    #ifdef __VERBOSE
        printf("w: %d h: %d, br: %d, if: %d, fr: %d\n", w, h, bitrate, iframeinterval, framerate);
    #endif

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
            printf("You can not use H265(HEVC) encoder because your libavcodec version is lower than 55.28.1.\n");
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
        printf("This videocodec doesn't support %s.\nThis videocodec support h264(AVC) and h265(HEVC)\n", codec);
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
        printf("This videocodec doesn't support %s.\nThis videocodec support rgb24 and bgr24\n", codec);
        #endif
        Py_RETURN_NONE;
    }

    #ifdef __VERBOSE
        printf("create VideoCodec class\n");
    #endif

    return PyCapsule_New((void*)new VideoCodec(codec_id, pixel_fmt, w, h, bitrate, iframeinterval, framerate), "ptr", destroy_video_codec);
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
    printf("packet size: %d\n", size);
    #endif

    memcpy( f->rgb[0], pkt, size);

    #ifdef __VERBOSE
    printf("copy success: %d\n", size);
    #endif

    // av_free_packet(&f->avpkt);
    // av_init_packet(&f->avpkt);
    av_packet_unref(&f->avpkt);
    // f->avpkt.data = NULL;    // packet data will be allocated by the encoder
    // f->avpkt.size = 0;

    #ifdef __VERBOSE
    printf("convert color format\n");
    #endif

    convertColorFormat(f);

    f->frame->pts = f->pts;
    f->pts+=1;

    int ret, got;

    ret = avcodec_encode_video2(f->mCodecContext, &f->avpkt, f->frame, &got);

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
        #ifdef __VERBOSE
        printf("encoded size: %d\n", f->avpkt.size);
        #endif
        return Py_BuildValue("y#", f->avpkt.data, f->avpkt.size);
    }
    // av_free_packet(&f->avpkt);
    Py_RETURN_NONE;
}

static PyMethodDef video_encoder_methods[] = {
        {"create_codec", create_video_codec, METH_VARARGS, "Create video codec. input argument (codec name, output pixel format). codec: h264 and h265 are supported. output pixel format: rgb24 and bgr24 are supported"},
        {"push_frame_data", push_frame_data, METH_VARARGS, "Push frame. and get encoded frame if available."},
        {NULL, NULL, 0, NULL} /* Sentinel */
    };

static struct PyModuleDef video_encoder_definition = { 
    PyModuleDef_HEAD_INIT,
    "videoencoder",
    "A Python module for ffmpeg video codec.",
    -1, 
    video_encoder_methods
};

PyMODINIT_FUNC PyInit_videoencoder(void) {
    Py_Initialize();
    return PyModule_Create(&video_encoder_definition);
}

#ifdef __cplusplus
 }
#endif
