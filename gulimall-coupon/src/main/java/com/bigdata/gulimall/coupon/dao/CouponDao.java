package com.bigdata.gulimall.coupon.dao;

import com.bigdata.gulimall.coupon.entity.CouponEntity;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * 优惠券信息
 *
 */
@Mapper
public interface CouponDao extends BaseMapper<CouponEntity> {
	
}
