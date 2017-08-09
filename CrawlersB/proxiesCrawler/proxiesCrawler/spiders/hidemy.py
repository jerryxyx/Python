# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from proxiesCrawler.items import ProxiescrawlerItem



class QuotesSpider(scrapy.Spider):
    name = "hidemy"
    def start_requests(self):
        pages = [i * 64 for i in range(0, 17)]
        urls = ["http://hidemy.name/en/proxy-list/?start="+str(startwith) for startwith in pages]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        address_list = response.xpath("//td[@class='tdl']/text()").extract()
        ports_list = response.xpath("//tr/td/text()").re('^\d{1,5}$')
                # symbol = response.xpath("//div[@class='qwidget-symbol']/text()").re_first(u'(\w+)\s')

        if(len(ports_list)!=len(address_list)):
            return
        proxies_list = [str(i)+':'+str(j) for (i,j) in zip(address_list,ports_list)]

        if(proxies_list):
            for proxy in proxies_list:
                item = ProxiescrawlerItem(proxies=proxy)
                yield item




# class HidemySpider(CrawlSpider):
#     name = 'hidemy'
#     allowed_domains = ['hidemy.name']
#     start_urls = ['http://hidemy.name/en/proxy-list']
#
#     rules = (
#         Rule(LinkExtractor(allow='\?start=.*/'), callback='parse_item', follow=True),
#     )
#
#     def parse_item(self, response):
#         address_list = response.xpath("//td[@class='tdl']/text()").extract()
#         ports_list = response.xpath("//tr/td/text()").re('^\d{1,4}$')
#         # symbol = response.xpath("//div[@class='qwidget-symbol']/text()").re_first(u'(\w+)\s')
#
#         if(len(ports_list)!=len(address_list) or len(ports_list)==0):
#
#             return
#         proxies_list = [str(i)+':'+str(j) for (i,j) in zip(address_list,ports_list)]
#
#         if(proxies_list):
#             item = ProxiescrawlerItem(proxies=proxies_list)
#
#         return item