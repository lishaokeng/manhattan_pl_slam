/*
 * patch_score.h
 *
 *  Created on: Dec 5, 2013
 *      Author: cforster
 */

#ifndef VIKIT_PATCH_SCORE_H_
#define VIKIT_PATCH_SCORE_H_

#include <stdint.h>

#if __SSE2__
#include <tmmintrin.h>
#endif

namespace vk {
namespace patch_score {


#if __SSE2__
// Horizontal sum of uint16s stored in an XMM register
inline int SumXMM_16(__m128i &target)
{
  unsigned short int sums_store[8];
  _mm_storeu_si128((__m128i*)sums_store, target); // 将128位值存入前一个参数, 分成16位一个数,无需16位对齐
  // 每个16位求和
  return sums_store[0] + sums_store[1] + sums_store[2] + sums_store[3] +
    sums_store[4] + sums_store[5] + sums_store[6] + sums_store[7];
}
// Horizontal sum of uint32s stored in an XMM register
inline int SumXMM_32(__m128i &target)
{
  unsigned int sums_store[4];
  _mm_storeu_si128((__m128i*)sums_store, target);
  return sums_store[0] + sums_store[1] + sums_store[2] + sums_store[3];
}
#endif


/********************************
 * @ function:  计算参考和当前patch的ZMSSD类
 *                            
 * @ param:     HALF_PATCH_SIZE    模板参数, 半个patch大小
 * 
 * @ note:      
 *******************************/
/// Zero Mean Sum of Squared Differences Cost
template<int HALF_PATCH_SIZE>
class ZMSSD {
public:

  static const int patch_size_ = 2*HALF_PATCH_SIZE;  //patch的大小
  static const int patch_area_ = patch_size_*patch_size_;  //patch的面积
  static const int threshold_  = 2000*patch_area_;  //
  uint8_t* ref_patch_;
  int sumA_, sumAA_;
//* 构造函数, 传入参考的patch
  ZMSSD(uint8_t* ref_patch) :
    ref_patch_(ref_patch)
  {
    uint32_t sumA_uint=0, sumAA_uint=0;
    for(int r = 0; r < patch_area_; r++)
    {
      uint8_t n = ref_patch_[r];
      sumA_uint += n; // patch内每个像素求和
      sumAA_uint += n*n; // patch内每个像素的平方和
    }
    sumA_ = sumA_uint;
    sumAA_ = sumAA_uint;
  }
//* 返回阈值大小
  static int threshold() { return threshold_; }
//* 在参考patch下,计算当前patch的得分
  int computeScore(uint8_t* cur_patch) const
  {
    uint32_t sumB_uint = 0;
    uint32_t sumBB_uint = 0;
    uint32_t sumAB_uint = 0;
    for(int r = 0; r < patch_area_; r++)
    {
      const uint8_t cur_pixel = cur_patch[r]; // 得到当前patch像素值
      sumB_uint  += cur_pixel;  // 当前patch像素求和
      sumBB_uint += cur_pixel*cur_pixel;  // 当前patch像素平方求和
      sumAB_uint += cur_pixel * ref_patch_[r];  // 当前patch像素乘以参考patch像素
    }
    const int sumB = sumB_uint; // ∑bi
    const int sumBB = sumBB_uint; // ∑bi²
    const int sumAB = sumAB_uint; // ∑ai*bi
    //! 公式: ∑(ai-bi)² - (∑ai-∑bi)² / all_pixel_num
    //* 零均值的差值平方和
    return sumAA_ - 2*sumAB + sumBB - (sumA_*sumA_ - 2*sumA_*sumB + sumB*sumB)/patch_area_;
  }
//* 不同参数的, 计算得分, 当前patch每隔stride取一行
  int computeScore(uint8_t* cur_patch, int stride) const
  {
    int sumB, sumBB, sumAB;
    // SSE2加速
#if __SSE2__
    if(patch_size_ == 8)
    {
      // From PTAM-GPL, Copyright 2008 Isis Innovation Limited
      __m128i xImageAsEightBytes;
      __m128i xImageAsWords;
      __m128i xTemplateAsEightBytes;
      __m128i xTemplateAsWords;
      __m128i xZero;
      //! 是16位的值求和,128个分成8份求和
      __m128i xImageSums;   // These sums are 8xuint16
      //! 是16位的乘积, 32位求和, 128位分成4份求和
      __m128i xImageSqSums; // These sums are 4xint32
      //! 同上
      __m128i xCrossSums;   // These sums are 4xint32
      __m128i xProduct;

      xImageSums = _mm_setzero_si128();
      xImageSqSums = _mm_setzero_si128();
      xCrossSums = _mm_setzero_si128();
      xZero = _mm_setzero_si128();

      uint8_t* imagepointer = cur_patch;
      uint8_t* templatepointer = ref_patch_;
      long unsigned int cur_stride = stride;
      // ! i表示整形
      //* 加载低64位, 高64位置为0
      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer); //加载cur的低64位(8个字节,8个像素)
      imagepointer += cur_stride; // 移动到下一个位置
      //* 两个参数的低64位分成每8位穿插着放[ a0,b0,a1,b1....a7,b7 ]
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero); //扩展成8个16位的
      //* 两个参数分为8个无符号的16位数相加
      //? 为什么以16位形式求和, 防止溢出?
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums); //得到cur的和
      
      //* r0 := (a0 * b0) + (a1 * b1)
      //* r1 := (a2 * b2) + (a3 * b3)
      //* r2 := (a4 * b4) + (a5 * b5)
      //* r3 := (a6 * b6) + (a7 * b7)
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords); //得到cur的平方
      //* 分为4个32位的有符号或无符号相加
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums); //得到平方和

      //* 参考的patch, 加载128位值
      xTemplateAsEightBytes=_mm_load_si128((__m128i*) templatepointer);//ref的16个字节(像素)
      templatepointer += 16; //移动到下16个
      xTemplateAsWords = _mm_unpacklo_epi8(xTemplateAsEightBytes,xZero); //ref的低8字节(64位)与0交叉,构成16位一个单位(words)
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords); //低8字节以words形式与cur的words相乘
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums); //求和
      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer); //加载cur的下8个字节
      imagepointer += cur_stride; // 下个位置
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero); // 构成cur的words
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums); // 求和
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords); // 求cur的平方
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums); //求平方和
      xTemplateAsWords = _mm_unpackhi_epi8(xTemplateAsEightBytes,xZero); //ref的高64位, 构成words
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords); //cur与ref交叉相乘
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums); //交叉积求和

      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer);
      imagepointer += cur_stride;
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero);
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums); //cur求和
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords);
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums); // cur平方和
      xTemplateAsEightBytes=_mm_load_si128((__m128i*) templatepointer);
      templatepointer += 16;
      xTemplateAsWords = _mm_unpacklo_epi8(xTemplateAsEightBytes,xZero);
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords); 
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums); //低64位交叉积求和
      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer);
      imagepointer += cur_stride;
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero);
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums);
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords);
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums);
      xTemplateAsWords = _mm_unpackhi_epi8(xTemplateAsEightBytes,xZero);
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords);
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums);

      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer);
      imagepointer += cur_stride;
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero);
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums);
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords);
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums);
      xTemplateAsEightBytes=_mm_load_si128((__m128i*) templatepointer);
      templatepointer += 16;
      xTemplateAsWords = _mm_unpacklo_epi8(xTemplateAsEightBytes,xZero);
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords);
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums);
      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer);
      imagepointer += cur_stride;
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero);
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums);
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords);
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums);
      xTemplateAsWords = _mm_unpackhi_epi8(xTemplateAsEightBytes,xZero);
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords);
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums);

      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer);
      imagepointer += cur_stride;
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero);
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums);
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords);
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums);
      xTemplateAsEightBytes=_mm_load_si128((__m128i*) templatepointer);
      templatepointer += 16;
      xTemplateAsWords = _mm_unpacklo_epi8(xTemplateAsEightBytes,xZero);
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords);
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums);
      xImageAsEightBytes=_mm_loadl_epi64((__m128i*) imagepointer);
      xImageAsWords = _mm_unpacklo_epi8(xImageAsEightBytes,xZero);
      xImageSums = _mm_adds_epu16(xImageAsWords,xImageSums);
      xProduct = _mm_madd_epi16(xImageAsWords, xImageAsWords);
      xImageSqSums = _mm_add_epi32(xProduct, xImageSqSums);
      xTemplateAsWords = _mm_unpackhi_epi8(xTemplateAsEightBytes,xZero);
      xProduct = _mm_madd_epi16(xImageAsWords, xTemplateAsWords);
      xCrossSums = _mm_add_epi32(xProduct, xCrossSums);

      //* 以上每一大块是16个字节(像素), 一共4块64个字节(像素),8*8的patch 
      sumB = SumXMM_16(xImageSums);
      sumAB = SumXMM_32(xCrossSums);
      sumBB = SumXMM_32(xImageSqSums);
    }
    else
#endif
// 这里#if 和  if 混用还挺有意思
    {
      uint32_t sumB_uint = 0;
      uint32_t sumBB_uint = 0;
      uint32_t sumAB_uint = 0;
      for(int y=0, r=0; y < patch_size_; ++y)
      {
        // 每隔stride(步长)取一行
        uint8_t* cur_patch_ptr = cur_patch + y*stride;
        for(int x=0; x < patch_size_; ++x, ++r)
        {
          const uint8_t cur_px = cur_patch_ptr[x];
          sumB_uint  += cur_px;
          sumBB_uint += cur_px * cur_px;
          sumAB_uint += cur_px * ref_patch_[r];
        }
      }
      sumB = sumB_uint;
      sumBB = sumBB_uint;
      sumAB = sumAB_uint;
    }
    return sumAA_ - 2*sumAB + sumBB - (sumA_*sumA_ - 2*sumA_*sumB + sumB*sumB)/patch_area_;
  }
};

} // namespace patch_score
} // namespace vk

#endif // VIKIT_PATCH_SCORE_H_
