package com.bigdata.gulimall.coupon.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.bigdata.common.utils.PageUtils;
import com.bigdata.gulimall.coupon.entity.CouponSpuRelationEntity;

import java.util.Map;

/**
 * 优惠券与产品关联
 *
 */
public interface CouponSpuRelationService extends IService<CouponSpuRelationEntity> {

    PageUtils queryPage(Map<String, Object> params);
}

