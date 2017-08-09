# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
import random
import re
import logging
import base64



log=logging.getLogger('scrapy.proxies')


class RandomUserAgent(object):
    def __init__(self,settings):
        self.agent_list=settings.get("AGENT_LIST")

    @classmethod
    def from_crawler(cls,crawler):
        return cls(crawler.settings)

    def process_request(self,request,spider):
        self.agents=[]
        for line in open(self.agent_list):
            self.agents.append(line)
        agency_used = random.choice(self.agents)
        request.headers['User-Agent']=agency_used
        log.info("***********************using agency: "+request.headers['User-Agent'].decode())

class Mode:
    RANDOMIZE_PROXY_EVERY_REQUESTS, RANDOMIZE_PROXY_ONCE, SET_CUSTOM_PROXY = range(3)


class RandomProxy(object):
    def __init__(self, settings):
        self.mode = settings.get('PROXY_MODE')
        self.proxy_list = settings.get('PROXY_LIST')
        self.chosen_proxy = ''
        if self.proxy_list is None:
            raise KeyError('PROXY_LIST setting is missing')

        if self.mode == Mode.RANDOMIZE_PROXY_EVERY_REQUESTS or self.mode == Mode.RANDOMIZE_PROXY_ONCE:
            fin = open(self.proxy_list)
            self.proxies = {}
            for line in fin.readlines():
                parts = re.match('(\w+://)?(\w+:\w+@)?(\d+\.\d+\.\d+\.\d+:\d+)', line.strip())
                if not parts:
                    continue

                # Cut trailing @
                if parts.group(2):
                    user_pass = parts.group(2)[:-1]
                else:
                    user_pass = ''

                if parts.group(1):
                    self.proxies[parts.group(1) + parts.group(3)] = user_pass
                else:
                    self.proxies['http://' + parts.group(3)] = user_pass


            fin.close()
            if self.mode == Mode.RANDOMIZE_PROXY_ONCE:
                self.chosen_proxy = random.choice(list(self.proxies.keys()))
        elif self.mode == Mode.SET_CUSTOM_PROXY:
            custom_proxy = settings.get('CUSTOM_PROXY')
            self.proxies = {}
            parts = re.match('(\w+://)?(\w+:\w+@)?(\d+\.\d+\.\d+\.\d+:\d+)', custom_proxy.strip())
            if not parts:
                raise ValueError('CUSTOM_PROXY is not well formatted')

            if parts.group(2):
                user_pass = parts.group(2)[:-1]
            else:
                user_pass = ''

            if parts.group(1):
                self.proxies[parts.group(1) + parts.group(3)] = user_pass
                self.chosen_proxy = parts.group(1) + parts.group(3)
            else:
                self.proxies['http://' + parts.group(3)] = user_pass
                self.chosen_proxy = 'http://' + parts.group(3)


    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_request(self, request, spider):
        # Don't overwrite with a random one (server-side state for IP)

        #retry! and delete the false proxy
        if 'proxy' in request.meta:
            #if request.meta["exception"] is False:
            log.info("proxy:"+request.meta['proxy'])
            log.info("url:"+request.url)
            if self.mode == Mode.RANDOMIZE_PROXY_EVERY_REQUESTS or self.mode == Mode.RANDOMIZE_PROXY_ONCE:
                proxy = request.meta['proxy']
                try:
                    del self.proxies[proxy]
                except KeyError:
                    pass
                #request.meta["exception"] = True
                if self.mode == Mode.RANDOMIZE_PROXY_ONCE:
                    self.chosen_proxy = random.choice(list(self.proxies.keys()))
                log.info('Removing failed proxy <%s>, %d proxies left' % (proxy, len(self.proxies)))
                # return
        #request.meta["exception"] = False
        if len(self.proxies) == 0:
            raise ValueError('All proxies are unusable, cannot proceed')

        if self.mode == Mode.RANDOMIZE_PROXY_EVERY_REQUESTS:
            proxy_address = random.choice(list(self.proxies.keys()))
        else:
            proxy_address = self.chosen_proxy


        proxy_user_pass = self.proxies[proxy_address]

        if proxy_user_pass:
            request.meta['proxy'] = proxy_address
            basic_auth = 'Basic ' + base64.b64encode(proxy_user_pass.encode()).decode()
            request.headers['Proxy-Authorization'] = basic_auth
        else:
            log.debug('Proxy user pass not found')
            request.meta['proxy'] = proxy_address
            #log.info(request.meta['proxy']+"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        #log.debug('Using proxy <%s>, %d proxies left' % (proxy_address, len(self.proxies)))

    def process_exception(self, request, exception, spider):
        log.info("Proxy exception********************************************")
        if 'proxy' not in request.meta:
            log.info("Proxy is not in request.meta********************************************")
            return
        if self.mode == Mode.RANDOMIZE_PROXY_EVERY_REQUESTS or self.mode == Mode.RANDOMIZE_PROXY_ONCE:
            proxy = request.meta['proxy']
            try:
                del self.proxies[proxy]
            except KeyError:
                pass
            request.meta["exception"] = True
            if self.mode == Mode.RANDOMIZE_PROXY_ONCE:
                self.chosen_proxy = random.choice(list(self.proxies.keys()))
            log.info('Removing failed proxy <%s>, %d proxies left' % (
                proxy, len(self.proxies)))
            #if self.mode == Mode.RANDOMIZE_PROXY_EVERY_REQUESTS:






class NsdqSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)
