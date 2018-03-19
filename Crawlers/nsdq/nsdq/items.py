# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
class GuruItem(scrapy.Item):
    symbol=scrapy.Field()
    #scorecard=scrapy.Field()
    PE_growth_PL=scrapy.Field()
    value_BG = scrapy.Field()
    momentum_strategy_V = scrapy.Field()
    growth_value_JO=scrapy.Field()
    small_cap_growth_MF=scrapy.Field()
    contrarian_DD = scrapy.Field()
    growth_value_MZ=scrapy.Field()
    price_sale_KF=scrapy.Field()


class NsdqItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass
