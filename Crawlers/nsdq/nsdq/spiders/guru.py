# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from nsdq.items import GuruItem
import pandas as pd
import re




class GuruSpider(CrawlSpider):
    name = 'guru'
    allowed_domains = ['nasdaq.com']
    #start_urls = ['http://nasdaq.com/']

    #rules = (
    #    Rule(LinkExtractor(allow=r'Items/'), callback='parse_item', follow=True),
    #)

    def start_requests(self):
        n_stock = getattr(self, 'n_stock', "100")
        #nsdqDF = pd.read_csv("http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=%s&render=download")
        #urls = nsdqDF.ix[:int(n_stock), 8]

        nsdqDF = pd.read_csv("/Users/xiayuxuan/PythonInFinance/Crawlers/resources/NASDAQ-100.csv")
        urls = ["http://www.nasdaq.com/symbol/"+symbol for symbol in nsdqDF.ix[:int(n_stock),0]]

        for url in urls:
            guru_url = url + "/guru-analysis"
            yield scrapy.http.Request(url=guru_url, callback=self.parse)

    def parse(self, response):
        # item = Myitem()
        percentage_list = response.xpath("//a[@href]/b/text()").extract()
        pattern = re.compile(r'\d{1,}')
        score_list = [int(pattern.match(i).group()) for i in percentage_list]
        #symbol = response.xpath("//div[@class='qwidget-symbol']/text()").re_first(u'(\w+)\s')
        symbol = response.url.split('/')[-2]
        if(score_list):

            item = GuruItem(symbol=symbol, PE_growth_PL=score_list[0],value_BG=score_list[1],
                            momentum_strategy_V=score_list[2],growth_value_JO=score_list[3],
                            small_cap_growth_MF=score_list[4],contrarian_DD=score_list[5],
                            growth_value_MZ=score_list[6],price_sale_KF=score_list[7])
        else:
            item = GuruItem(symbol=symbol)
        return item