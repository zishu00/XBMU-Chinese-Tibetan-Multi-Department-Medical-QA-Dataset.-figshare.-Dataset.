import requests
import time
import json
import csv
import re
from bs4 import BeautifulSoup
import random
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime

class DXYDataCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 科室映射（根据丁香园实际分类调整）
        self.department_mapping = {
            'ent': '耳鼻喉科',
            'ophthalmology': '眼科',
            'internal': '内科',
            'neurology': '神经内科',
            'surgery': '外科',
            'nutrition': '营养保健科'
        }
        
        self.base_url = "https://dxy.com"
        self.data = []
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dxy_crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_question_list(self, department, page=1):
        """获取问题列表页"""
        try:
            # 根据科室构建URL（需要根据丁香园实际URL结构调整）
            if department in self.department_mapping:
                url = f"{self.base_url}/faq/{department}?page={page}"
            else:
                url = f"{self.base_url}/faq?department={department}&page={page}"
                
            self.logger.info(f"正在爬取: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            questions = []
            
            # 解析问题列表（选择器需要根据实际页面结构调整）
            question_items = soup.select('.question-item, .faq-item, .list-item')  # 可能需要调整
            
            for item in question_items:
                try:
                    question_link = item.select_one('a[href*="/question/"], a[href*="/faq/"]')
                    if question_link:
                        title = question_link.get_text(strip=True)
                        link = urljoin(self.base_url, question_link.get('href'))
                        questions.append({
                            'title': title,
                            'link': link,
                            'department': self.department_mapping.get(department, department)
                        })
                except Exception as e:
                    self.logger.warning(f"解析问题项失败: {e}")
                    continue
                    
            return questions
            
        except Exception as e:
            self.logger.error(f"获取问题列表失败: {e}")
            return []

    def get_question_detail(self, question_url):
        """获取问题详情页的内容"""
        try:
            self.logger.info(f"正在获取问题详情: {question_url}")
            
            response = self.session.get(question_url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取问题（选择器需要根据实际页面结构调整）
            question_elem = soup.select_one('.question-content, .question-text, .faq-question')
            question = question_elem.get_text(strip=True) if question_elem else ""
            
            # 提取回答（选择器需要根据实际页面结构调整）
            answer_elem = soup.select_one('.answer-content, .doctor-answer, .faq-answer')
            answer = answer_elem.get_text(strip=True) if answer_elem else ""
            
            # 提取症状背景信息（如果有）
            context_elem = soup.select_one('.symptoms, .context, .patient-info')
            context = context_elem.get_text(strip=True) if context_elem else ""
            
            # 如果无法直接提取，尝试其他选择器
            if not question:
                question_elem = soup.find('h1') or soup.find('h2')
                question = question_elem.get_text(strip=True) if question_elem else ""
                
            if not answer:
                # 尝试提取医生回答部分
                answer_elems = soup.select('.doctor-reply, .expert-answer, .medical-advice')
                if answer_elems:
                    answer = ' '.join([elem.get_text(strip=True) for elem in answer_elems])
            
            return {
                'question': question,
                'answer': answer,
                'context': context,
                'url': question_url
            }
            
        except Exception as e:
            self.logger.error(f"获取问题详情失败 {question_url}: {e}")
            return None

    def crawl_by_department(self, department, max_pages=10):
        """按科室爬取数据"""
        self.logger.info(f"开始爬取 {department} 科室数据")
        
        all_questions = []
        
        for page in range(1, max_pages + 1):
            questions = self.get_question_list(department, page)
            if not questions:
                self.logger.info(f"{department} 第{page}页无数据，停止爬取")
                break
                
            all_questions.extend(questions)
            self.logger.info(f"{department} 第{page}页获取到 {len(questions)} 个问题")
            
            # 随机延时，避免请求过快
            time.sleep(random.uniform(1, 3))
        
        # 获取问题详情
        successful_count = 0
        for i, q in enumerate(all_questions):
            detail = self.get_question_detail(q['link'])
            if detail and detail['question'] and detail['answer']:
                data_item = {
                    'id': f"{department}_{i+1}",
                    'question_zh': detail['question'],
                    'answer_zh': detail['answer'],
                    'context': detail['context'],
                    'category': q['department'],
                    'source_url': q['link'],
                    'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.data.append(data_item)
                successful_count += 1
                self.logger.info(f"成功获取 {successful_count}/{len(all_questions)} 个问题详情")
            
            # 随机延时
            time.sleep(random.uniform(0.5, 2))
            
            # 每10个问题保存一次，防止数据丢失
            if successful_count % 10 == 0:
                self.save_to_csv(f'dxy_data_partial_{department}.csv')
        
        self.logger.info(f"{department} 科室爬取完成，成功获取 {successful_count} 个问答对")
        return successful_count

    def crawl_all_departments(self, max_pages_per_dept=5):
        """爬取所有科室的数据"""
        total_count = 0
        
        for dept in self.department_mapping.keys():
            count = self.crawl_by_department(dept, max_pages_per_dept)
            total_count += count
            
            # 科室间较长延时
            time.sleep(random.uniform(5, 10))
        
        self.logger.info(f"所有科室爬取完成，共获取 {total_count} 个问答对")
        return total_count

    def save_to_csv(self, filename='dxy_medical_qa.csv'):
        """保存数据到CSV文件"""
        if not self.data:
            self.logger.warning("没有数据可保存")
            return
            
        fieldnames = ['id', 'question_zh', 'answer_zh', 'context', 'category', 'source_url', 'crawl_time']
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data)
                
            self.logger.info(f"数据已保存到 {filename}, 共 {len(self.data)} 条记录")
            
        except Exception as e:
            self.logger.error(f"保存CSV文件失败: {e}")

    def save_to_json(self, filename='dxy_medical_qa.json'):
        """保存数据到JSON文件"""
        if not self.data:
            self.logger.warning("没有数据可保存")
            return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"数据已保存到 {filename}, 共 {len(self.data)} 条记录")
            
        except Exception as e:
            self.logger.error(f"保存JSON文件失败: {e}")

def main():
    """主函数"""
    crawler = DXYDataCrawler()
    
    try:
        # 爬取所有科室，每科室最多3页（测试用，实际可增加）
        total_count = crawler.crawl_all_departments(max_pages_per_dept=3)
        
        if total_count > 0:
            # 保存数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'dxy_medical_qa_{timestamp}.csv'
            json_filename = f'dxy_medical_qa_{timestamp}.json'
            
            crawler.save_to_csv(csv_filename)
            crawler.save_to_json(json_filename)
            
            print(f"\n爬取完成！共获得 {total_count} 个医疗问答对")
            print(f"数据已保存到: {csv_filename} 和 {json_filename}")
        else:
            print("没有获取到数据，请检查网络连接或网站结构是否变化")
            
    except KeyboardInterrupt:
        print("\n用户中断爬取")
        if crawler.data:
            crawler.save_to_csv('dxy_medical_qa_interrupted.csv')
    except Exception as e:
        print(f"爬取过程中发生错误: {e}")
        if crawler.data:
            crawler.save_to_csv('dxy_medical_qa_error.csv')

if __name__ == "__main__":
    main()