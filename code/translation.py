import requests
import re
import json
import subprocess
import os
import sys
import argparse
from time import time
import sentencepiece as spm
from sacrebleu import corpus_bleu
import pdb
'''
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
'''
class Translation():

    #类初始化：初始化翻译类，加载分词模型，定义HTTP请求模板、语言代码映射及分句参数。
    def __init__(self):
        # 加载SentencePiece分词模型
        self.sp_tok = spm.SentencePieceProcessor("./spm.model")
        # HTTP请求模板
        self.http_content = 'curl -i -X POST -H "Content-Type: application/json" -d {user_data} http://210.26.8.106:60006/translate'

        #------------------------------------------------------------
        # 语言代码映射
        self.lang_idx = {'藏语':'TB',
                    '蒙古语':'MN',
                    '维吾尔语':'UG',
                    '印地语':'HI',
                    '越南语':'VI',
                    '汉语':'ZH',}
        # 单句最大分词数
        self.filter_max_seq=32
        # 段落最大分词数
        self.filter_max_sent_seq=128

    #执行Shell命令：执行Shell命令并返回状态码和输出结果。
    def get_shell(self, cmd):
        result = subprocess.getstatusoutput(cmd)
        return result
    
    #获取语言代码：根据语言名称返回对应的缩写代码（如“汉语”→“ZH”）。
    def get_language_index(self, language):
        
        language = language.strip()
        if language == "" or language not in self.lang_idx:
            return None
        return self.lang_idx[language.strip()]

    #分词：对输入文本进行分词，返回子词列表。
    def tokenize(self, line):
        return self.sp_tok.encode(line, out_type=str)

    #分块检查：将段落按分词长度分块，确保每块不超过max_seq。
    def check_segments(self, val, max_seq, src_lan="ZH", tgt_lan="TB"):
        # 遍历段落和句子，按max_seq限制合并或拆分句子
        # 返回分块后的句子列表
        if len(val) == 0:
            return val
        seg_list = []
        for i in range(len(val)):
            cur_sent = ""
            cur_size = 0
            sents_list = []
            #pdb.set_trace()
            for j in range(len(val[i])):
                str_ = val[i][j].strip()
                tokens = self.tokenize(str_)
                if len(tokens) > self.filter_max_sent_seq:
                    continue
                if (cur_size + len(tokens)) > max_seq:
                    if cur_sent != "":
                        sents_list.append(cur_sent)
                    cur_sent = str_
                    cur_size = len(tokens)
                else:
                    cur_sent += (str_ + " ")
                    cur_size += len(tokens)
            if cur_sent != "":
                sents_list.append(cur_sent.strip())
            seg_list.append(sents_list.copy())

        return seg_list

    #添加标签：为每个句子添加XML风格的标签，用于标识翻译方向（如<ZH-TB>）。
    def append_list_tag(self, val_lists, src_lan, tgt_lan):
        # 为每个句子添加<源语言-目标语言>标签
        # 示例：<ZH-TB>句子内容</ZH-TB>
        head = "<"+src_lan+"-"+tgt_lan+">"
        tail = "</"+src_lan+"-"+tgt_lan+">"
        for i in range(len(val_lists)):
            for j in range(len(val_lists[i])):
                line = val_lists[i][j]
                line = head + line
                val_lists[i][j] = line
        return val_lists

    #语言格式化方法：功能：根据不同语言的分隔符（如中文“。”、藏语“།”）分句，再分块并添加标签。
    #藏语
    def format_Tibetan(self, text, src_lan, tgt_lan, append_tag=True):
        #pdb.set_trace()
        seg_list = []
        text = text.strip()
        if text == "":
            return seg_list
        for line in text.split("\n"):
            line = line.strip()
            if line == "":
                continue
            sents_list = []
            if len(self.tokenize(line)) < self.filter_max_seq:
                sents_list.append(line)
            else:
                segs = line.split("།")
                for idx in range(len(segs)):
                    sent = segs[idx]
                    sent = sent.strip()
                    if sent == "":
                        continue
                    if (idx < len(segs)-1):
                        sent += "།"
                    sents_list.append(sent)
            seg_list.append(sents_list)
        seg_list = self.check_segments(seg_list, self.filter_max_seq)
        if append_tag:
            seg_list = self.append_list_tag(seg_list, src_lan, tgt_lan)
        
        return seg_list
    #中文
    def format_Chinese(self, text, src_lan, tgt_lan, append_tag=True):
        #pdb.set_trace()
        seg_list = []
        text = text.strip()
        if text == "":
            return seg_list
        for line in text.split("\n"):
            line = line.strip()
            if line == "":
                continue
            sents_list = []
            if len(self.tokenize(line)) < self.filter_max_seq:
                sents_list.append(line)
            else:
                segs = line.split("。")
                for idx in range(len(segs)):
                    sent = segs[idx]
                    sent = sent.strip()
                    if sent == "":
                        continue
                    if (idx < len(segs)-1):
                        sent += "。"
                    sents_list.append(sent)
            seg_list.append(sents_list)
        seg_list = self.check_segments(seg_list, self.filter_max_seq)
        if append_tag:
            seg_list = self.append_list_tag(seg_list, src_lan, tgt_lan)
        
        return seg_list
    #维吾尔语
    def format_Uyghur(self, text, src_lan, tgt_lan, append_tag=True):
        #pdb.set_trace()
        seg_list = []
        text = text.strip()
        if text == "":
            return seg_list
        for line in text.split("\n"):
            line = line.strip()
            if line == "":
                continue
            sents_list = []
            if len(self.tokenize(line)) < self.filter_max_seq:
                sents_list.append(line)
            else:
                segs = line.split(".")
                for idx in range(len(segs)):
                    sent = segs[idx]
                    sent = sent.strip()
                    if sent == "":
                        continue
                    if (idx < len(segs)-1):
                        sent += "."
                    sents_list.append(sent)
            seg_list.append(sents_list)
        seg_list = self.check_segments(seg_list, self.filter_max_seq)
        if append_tag:
            seg_list = self.append_list_tag(seg_list, src_lan, tgt_lan)
        
        return seg_list
    #蒙古语
    def format_Mongolian(self, text, src_lan, tgt_lan, append_tag=True):
        #pdb.set_trace()
        seg_list = []
        text = text.strip()
        if text == "":
            return seg_list
        for line in text.split("\n"):
            line = line.strip()
            if line == "":
                continue
            sents_list = []
            if len(self.tokenize(line)) < self.filter_max_seq:
                sents_list.append(line)
            else:
                segs = line.split("᠃")
                for idx in range(len(segs)):
                    sent = segs[idx]
                    sent = sent.strip()
                    if sent == "":
                        continue
                    if (idx < len(segs)-1):
                        sent += "᠃"
                    sents_list.append(sent)
            seg_list.append(sents_list)
        seg_list = self.check_segments(seg_list, self.filter_max_seq)
        if append_tag:
            seg_list = self.append_list_tag(seg_list, src_lan, tgt_lan)
        
        return seg_list

##'''
    #获取翻译：发送HTTP请求获取翻译结果，解析返回的JSON格式数据，提取翻译内容。
    def get_trans_sl(self, sent_list, src_lan, tgt_lan, append_tag):
    ##sent_list：待翻译的文本列表（单个句子或多个句子组成的列表）。
    ##src_lan：源语言代码（如 "TB" 表示藏语）。
    ##tgt_lan：目标语言代码（如 "ZH" 表示汉语）。
    ##append_tag：布尔值，控制是否移除句子中的 XML 标签（如 <TB-ZH>）。
        ##确保输入 sent_list 为列表类型。
        if type(sent_list) == str:
            sent_list = [sent_list]
        elif type(sent_list) != list:
            print("仅支持list数据")
            return None
        
        #构建 HTTP 请求的 JSON 数据模板
        buf_head = "{\"src\": \""##buf_head：JSON 对象的头部，格式为 {"src": "。
        buf_tail = "\", \"id\": 1}"##buf_tail：JSON 对象的尾部，格式为 ", "id": 1}。
        user_format = "\'[{buffer_}]\'"##user_format：最终 HTTP 请求数据的模板，格式为 '[{buffer_}]'。
        user_buffer = ""##user_buffer：动态生成的 JSON 数据内容。
        
        #遍历句子列表，拼接 JSON 数据
        for idx in range(len(sent_list)):
            sent = sent_list[idx]
            item = buf_head + sent + buf_tail
            user_buffer += item
            if (idx < (len(sent_list)) - 1):
                user_buffer += ","
        
        user_format = user_format.format(buffer_=user_buffer)
        cmd = self.http_content.format(user_data=user_format)
        response = self.get_shell(cmd)
        contents = response[1].split("Server: waitress")[1].strip()
        if contents.startswith("[["):
            contents = contents[2:]
        if contents.endswith("]]"):
            contents = contents[:-2]
        json_items = contents.split(',{"n_best":1,')
        res = ""
        for line in json_items:
            line = line.strip()
            #pdb.set_trace()
            if line == "":
                continue
            line = line.replace("\n","")
            try:
                line_ = line.split('"tgt":"')[-1]
                line_ = line_[:-2]
                val = line_
                val = val.encode('utf-8').decode('unicode_escape')
                #pdb.set_trace()
                #val = re.search("[\\u4e00-\\u9fa5]+", val)[0]
                val = val.replace(" ", "")
                val = val.replace("▁", " ")
                val = val.replace(",", "，")
                val = val.replace(";", "；")
                #val = val.replace("“", "\u201c")
                #val = val.replace("”", "\u201d")
                if append_tag:
                    #pdb.set_trace()
                    val = val.replace("<"+src_lan+"-"+tgt_lan+">", "")
                    #val = val.replace("</"+src_lan+"-"+tgt_lan+">", "")
                val = val.strip()
            except:
            #    pdb.set_trace()
                val = "<Decoder Error>"
            res += (val)
        
        return res.strip()
##'''    
    def get_trans(self, sent_list, src_lan, tgt_lan, append_tag):

        # 输入类型检查与请求构造（保持不变）
        if not isinstance(sent_list, list):
            sent_list = [sent_list] if isinstance(sent_list, str) else []
        
        data = [{"src": sent, "id": 1} for sent in sent_list]
        ##data = [{"src": sent, "id": idx} for idx, sent in enumerate(sent_list)]
        try:
            user_data = json.dumps(data, ensure_ascii=False)
        except Exception as e:
            print(f"JSON序列化失败: {e}")
            return "<JSON Error>"

        # 发送HTTP请求（保持不变）
        url = "http://210.26.8.106:60006/translate"
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, data=user_data.encode("utf-8"))
            response.raise_for_status()
            json_response = response.json()
            print("[DEBUG] 原始响应结构:", json_response)
        except Exception as e:
            print(f"HTTP请求失败: {e}")
            return "<HTTP Error>"

        # 解析响应（关键修改：移除unicode_escape解码）
        translated_texts = []
        try:
            for outer_list in json_response:
                if not isinstance(outer_list, list):
                    continue
                for item in outer_list:
                    tgt_text = item.get("tgt", "")
                    # 直接使用原始文本，无需转码
                    tgt_text = tgt_text.replace("▁", " ").strip()  # 清理SentencePiece符号
                    if append_tag:
                        tag = f"<{src_lan}-{tgt_lan}>"
                        tgt_text = tgt_text.replace(tag, "").strip()
                    translated_texts.append(tgt_text)
        except Exception as e:
            print(f"解析失败: {e}")
            return "<Decoder Error>"

        return " ".join(translated_texts)
    
    def get_trans_s(self, sent_list, src_lan, tgt_lan, append_tag=True):
        if not sent_list:
            return ""
        
        # 构建请求数据
        data = [{"src": sent, "id": idx} for idx, sent in enumerate(sent_list)]
        try:
            payload = json.dumps(data, ensure_ascii=False)
        except Exception as e:
            print(f"[JSON错误] 序列化失败: {e}")
            return "<JSON Error>"
        
        # 配置请求参数
        url = "http://210.26.8.106:60006/translate"
        headers = {"Content-Type": "application/json"}
        
        # 配置重试策略
        retries = Retry(
            total=2,  # 减少重试次数
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        session = requests.Session()
        session.mount("http://", HTTPAdapter(max_retries=retries))
        
        try:
            # 发送请求（延长超时时间）
            response = session.post(
                url,
                headers=headers,
                data=payload.encode("utf-8"),
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # 打印详细错误信息
            if hasattr(e, "response") and e.response is not None:
                print(f"[HTTP错误] 状态码: {e.response.status_code}")
                print(f"[DEBUG] 错误响应: {e.response.text[:200]}")  # 截断长文本
            else:
                print(f"[HTTP错误] 请求失败: {str(e)}")
            return "<HTTP Error>"

        # 解析响应（处理多层嵌套结构）
        try:
            raw_response = response.json()
            print(f"[DEBUG] 原始响应结构: {raw_response}")  # 调试用

            # 递归展开嵌套列表
            def unpack_nested(data):
                if isinstance(data, list):
                    for item in data:
                        yield from unpack_nested(item)
                elif isinstance(data, dict):
                    yield data
                    
            translated_texts = []
            for item in unpack_nested(raw_response):
                # 提取翻译文本
                tgt_text = item.get("tgt", "")
                
                # 清理格式
                tgt_text = (
                    tgt_text.replace("▁", " ")
                    .replace("\n", " ")
                    .strip()
                )
                
                # 移除XML标签
                if append_tag:
                    open_tag = f"<{src_lan}-{tgt_lan}>"
                    close_tag = f"</{src_lan}-{tgt_lan}>"
                    tgt_text = tgt_text.replace(open_tag, "").replace(close_tag, "")
                    
                translated_texts.append(tgt_text)
                
            return " ".join(translated_texts)

        except Exception as e:
            print(f"[解析错误] 响应解析失败: {e}")
            print(f"[DEBUG] 原始响应内容:\n{response.text}")
            return "<Decoder Error>"
##'''
    #翻译方法：根据用户输入的文本、源语言和目标语言，调用相应的格式化方法进行分句和分块处理，然后获取翻译结果。
    def translate_sl(self, user_input, src_lan="ZH", tgt_lan="TB", append_tag=True):
        #pdb.set_trace()
        text = user_input.strip()
        if text == "":
            return text
        
        val = []
        if src_lan == "ZH":
            val = self.format_Chinese(user_input, src_lan, tgt_lan, append_tag)
        elif src_lan == "TB":
            val = self.format_Tibetan(user_input, src_lan, tgt_lan, append_tag)
        elif src_lan == "MN":
            val = self.format_Mongolian(user_input, src_lan, tgt_lan, append_tag)
        elif src_lan == "UG":
            val = self.format_Uyghur(user_input, src_lan, tgt_lan, append_tag)
        else:
            return "选择语言不支持"
        text_buffer = ""
        seq_size = 2048
        for i in range(len(val)):
            buffer_len = 0
            buffer_list = []
            for sent in val[i]:
                res = ""
                if buffer_len > seq_size:
                    try:
                        res = self.get_trans(buffer_list, src_lan, tgt_lan, append_tag)
                    except:
                        res = "<Batch Error>"
                    text_buffer += res
                    buffer_list.clear()
                    buffer_list.append(sent)
                    buffer_len = len(sent)
                else:
                    buffer_list.append(sent)
                    buffer_len += len(sent)
            if (len(buffer_list) > 0):
                #pdb.set_trace()
                try:
                    res = self.get_trans(buffer_list, src_lan, tgt_lan, append_tag)
                except:
                    res = "<Batch Error>"
                text_buffer += res
            if i < len(val):
                text_buffer += "\n"
        return text_buffer
##'''
    
    def translate_s(self, user_input, src_lan="ZH", tgt_lan="TB", append_tag=True):
        text = user_input.strip()
        if not text:
            return text

        # 获取分块后的句子列表
        val = []
        if src_lan == "ZH":
            val = self.format_Chinese(text, src_lan, tgt_lan, append_tag)
        # 其他语言分支省略...

        text_buffer = []
        for paragraph in val:
            batch = []
            current_token_count = 0

            for sentence in paragraph:
                tokens = self.tokenize(sentence)
                token_count = len(tokens)

                # 如果当前句子超过最大长度，跳过或拆分（根据需求调整）
                if token_count > self.filter_max_seq:
                    continue

                # 累积分块
                if current_token_count + token_count > self.filter_max_seq:
                    translated = self.get_trans(batch, src_lan, tgt_lan, append_tag)
                    text_buffer.append(translated)
                    batch = []
                    current_token_count = 0

                batch.append(sentence)
                current_token_count += token_count

            # 处理剩余句子
            if batch:
                translated = self.get_trans(batch, src_lan, tgt_lan, append_tag)
                text_buffer.append(translated)

        return "\n".join(text_buffer)
    
    def translate(self, user_input, src_lan="ZH", tgt_lan="TB", append_tag=True):
        text = user_input.strip()
        if not text:
            return text

        # 获取分块后的句子列表（需根据语言调用对应的格式化方法）
        if src_lan == "ZH":
            formatted_segments = self.format_Chinese(text, src_lan, tgt_lan, append_tag)
        elif src_lan == "TB":
            formatted_segments = self.format_Tibetan(text, src_lan, tgt_lan, append_tag)
        # 其他语言分支省略...

        translated_text = []
        for paragraph in formatted_segments:
            batch = []
            current_length = 0

            # 分块逻辑：基于分词数而非字符数
            for sentence in paragraph:
                tokens = self.tokenize(sentence)
                token_count = len(tokens)
                
                # 若当前句子超过最大长度，直接跳过（或按需拆分）
                ##if token_count > self.filter_max_seq:
                    ##continue
                
                # 累积至阈值时发送翻译请求
                if current_length + token_count > self.filter_max_seq:
                    try:
                        res = self.get_trans(batch, src_lan, tgt_lan, append_tag)
                        translated_text.append(res)
                    except Exception as e:
                        print(f"分块翻译失败: {e}")
                        translated_text.append("<Translation Error>")
                    batch = []
                    current_length = 0
                
                batch.append(sentence)
                current_length += token_count

            # 处理剩余句子
            if batch:
                try:
                    res = self.get_trans(batch, src_lan, tgt_lan, append_tag)
                    translated_text.append(res)
                except Exception as e:
                    print(f"分块翻译失败: {e}")
                    translated_text.append("<Translation Error>")

        # 合并段落并保留原始换行
        return "\n".join(translated_text)