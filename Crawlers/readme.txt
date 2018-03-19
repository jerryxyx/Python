conda info -e

source activate py36

scrapy startproject myproject
scrapy genspider name mydomain.com
scrapy genspider -t crawl scrapyname domain.com

scrapy crawl myspider
scrapy list
scrapy fetch url
scrapy shell url

About selector
response.xpath("//div/td[@class="hh"]/text()").re('\d+\.\d+")

About conda
conda info -e
source activate py36

About nsdq
scrapy crawl guru -o guru1.csv -a n_stock=100

About proxiesCrawler
scrapy crawl hidemy -o proxies1.csv

About git
git clone https://jerryxyx/Crawlers.git
git add .
git add -u
git commit -m "commit"
git push origin master
git pull origin master
