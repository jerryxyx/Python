# -*- coding: utf-8 -*-
import scrapy


class ProxyspiderSpider(scrapy.Spider):
    name = "proxySpider"
    allowed_domains = ["google.com"]
    start_urls = ['http://google.com/']

    def parse(self, response):
        pass
