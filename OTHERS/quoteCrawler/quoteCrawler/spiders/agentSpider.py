# -*- coding: utf-8 -*-
#import scrapy
from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
from quoteCrawler.items import CnstockItem


class AgentspiderSpider(CrawlSpider):
    name = "agentSpider"
    allowed_domains = ["news.cnstock.com"]
    start_urls = ['http://news.cnstock.com/news/sns_yw/index.html/']
    rules = [
            Rule(LinkExtractor(allow=("/sns_yw/\d+$")),follow=True),
            Rule(LinkExtractor(allow=("/.+,.*")),callback='parse_2'),
            ]
    def parse_2(self,response):
        self.logger.info("from self logger**************")
        item=CnstockItem()
        item['url']=response.url
        item['title']=response.xpath("//h1[@class='title']/text()").extract_first()
        yield item
       #yield {
        #        "url":response.url,
         #       "title":response.xpath("//h1[@class='title']/text()").extract_first()
          #      }




