# -*- coding: utf-8 -*-

from scrapy import Request, Spider

class Scraper(Spider):
    name = u'scraper'

    def start_requests(self):
        """This is our first request to grab all the urls of the profiles.
        """
        yield Request(
            url=u'http://scraping-challenge-2.herokuapp.com',
            callback=self.parse,
        )

    def parse(self, response):
        """We have all the urls of the profiles. Let's make a request for each profile.
        """
        urls = response.xpath(u'//a/@href').extract()
        for url in urls:
            yield Request(
                url=response.urljoin(url),
                callback=self.parse_profile,
            )

    def parse_profile(self, response):
        """We have a profile. Let's extract the name
        """
        name_el = response.css(u'.profile-info-name::text').extract()
        if len(name_el) > 0:
            yield {
                'name': name_el[0]
            }