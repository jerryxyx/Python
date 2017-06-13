# -*- coding: utf-8 -*-
import scrapy
import re
import pandas as pd
from nsdqCrawler.items import GuruItem


class GuruSpider(scrapy.Spider):
    name = "guru"
    allowed_domains = ["nasdaq.com"]
    #start_urls = ['http://nasdaq.com/']

    def start_requests(self):
        n_stock = getattr(self,'n_stock',2)
        nsdqDF=pd.read_csv("http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=%s&render=download")
        urls=nsdqDF.ix[:n_stock,8]
        #urls=nsdqDF.ix[:2,8]
        for url in urls:
            guru_url=url+"/guru-analysis"
            yield scrapy.http.FormRequest(url=guru_url,callback=self.parse)

    def parse(self, response):
        #item = Myitem()
        percentage_list=response.xpath("//a[@href]/b/text()").extract()
        pattern=re.compile(r'\d{1,}')
        score_list=[int(pattern.match(i).group()) for i in percentage_list]
        symbol=response.xpath("//div[@class='qwidget-symbol']/text()").re_first(u'(\w+)\s')
        item = GuruItem(symbol=symbol,scores=score_list)
        yield item
        #yield{
        #         'symbol':symbol,
         #        'scores':score_list
          #       }



