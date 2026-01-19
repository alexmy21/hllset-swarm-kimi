import os
import re
from pathlib import Path
from typing import List, Optional, Set
from collections import Counter

# ------------------------------------------------------------
# Advanced Corpus Loader with Cleaning Options
# ------------------------------------------------------------

class ChineseCorpusLoader:
    """Advanced loader for Chinese text corpus"""
    
    def __init__(self, 
                 min_length: int = 3,
                 max_length: int = 100,
                 remove_duplicates: bool = True,
                 split_on_punctuation: bool = False,
                 punctuation_marks: str = '。！？；.,!?;'):
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.split_on_punctuation = split_on_punctuation
        self.punctuation_marks = punctuation_marks
        self.stats = {
            'files_read': 0,
            'lines_processed': 0,
            'chars_filtered': 0,
            'sentences_final': 0
        }
    
    @staticmethod
    def is_chinese_char(char: str) -> bool:
        """Check if character is Chinese"""
        code = ord(char)
        return (0x4E00 <= code <= 0x9FFF or    # CJK Unified
                0x3400 <= code <= 0x4DBF or    # CJK Extension A
                0x20000 <= code <= 0x2A6DF or  # CJK Extension B
                0xF900 <= code <= 0xFAFF)      # CJK Compatibility
    
    def filter_chinese(self, text: str) -> str:
        """Keep only Chinese characters"""
        filtered = ''.join(char for char in text if self.is_chinese_char(char))
        self.stats['chars_filtered'] += len(text) - len(filtered)
        return filtered
    
    def split_by_punctuation(self, text: str) -> List[str]:
        """Split text by Chinese punctuation marks"""
        if not self.split_on_punctuation:
            return [text]
        
        # Create regex pattern for punctuation
        pattern = f"[{re.escape(self.punctuation_marks)}]"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def load_file(self, file_path: str) -> List[str]:
        """Load single file with encoding detection"""
        encodings = ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"  ✓ {Path(file_path).name}: {encoding}")
                self.stats['files_read'] += 1
                return content.split('\n')
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        print(f"  ✗ Could not read {file_path}")
        return []
    
    def process_lines(self, lines: List[str]) -> List[str]:
        """Process lines into valid sentences"""
        sentences = []
        
        for line in lines:
            self.stats['lines_processed'] += 1
            
            # Filter to Chinese only
            chinese_text = self.filter_chinese(line.strip())
            
            if not chinese_text:
                continue
            
            # Optionally split by punctuation
            parts = self.split_by_punctuation(chinese_text)
            
            for part in parts:
                # Check length constraints
                if self.min_length <= len(part) <= self.max_length:
                    sentences.append(part)
                elif len(part) > self.max_length:
                    # Split long sentences into chunks
                    for i in range(0, len(part), self.max_length):
                        chunk = part[i:i+self.max_length]
                        if len(chunk) >= self.min_length:
                            sentences.append(chunk)
        
        return sentences
    
    def load_corpus(self, file_paths: List[str]) -> List[str]:
        """Load entire corpus from multiple files"""
        corpus = []
        
        print(f"Loading {len(file_paths)} files...")
        print("="*60)
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"  ✗ Not found: {file_path}")
                continue
            
            lines = self.load_file(file_path)
            sentences = self.process_lines(lines)
            corpus.extend(sentences)
            
            print(f"    → {len(sentences)} sentences")
        
        # Remove duplicates if requested
        if self.remove_duplicates:
            before = len(corpus)
            seen = set()
            corpus = [x for x in corpus if not (x in seen or seen.add(x))]
            print(f"\nRemoved {before - len(corpus)} duplicates")
        
        self.stats['sentences_final'] = len(corpus)
        self.print_stats(corpus)
        
        return corpus
    
    def print_stats(self, corpus: List[str]):
        """Print corpus statistics"""
        if not corpus:
            print("\n⚠ Empty corpus!")
            return
        
        all_chars = ''.join(corpus)
        char_freq = Counter(all_chars)
        
        print(f"\n{'='*60}")
        print(f"Corpus Statistics:")
        print(f"  Files read: {self.stats['files_read']}")
        print(f"  Lines processed: {self.stats['lines_processed']}")
        print(f"  Non-Chinese chars filtered: {self.stats['chars_filtered']}")
        print(f"  Final sentences: {self.stats['sentences_final']}")
        print(f"  Total characters: {len(all_chars):,}")
        print(f"  Unique characters: {len(char_freq)}")
        print(f"  Avg sentence length: {len(all_chars) / len(corpus):.1f}")
        print(f"  Min/Max length: {min(len(s) for s in corpus)} / {max(len(s) for s in corpus)}")
        
        # Top 20 most common characters
        print(f"\nTop 20 characters:")
        for char, count in char_freq.most_common(20):
            print(f"  {char}: {count:,}")
        
        # Sample sentences
        print(f"\nSample sentences:")
        for i, sent in enumerate(corpus[:10]):
            print(f"  {i+1}. {sent} ({len(sent)} chars)")
        if len(corpus) > 10:
            print(f"  ... and {len(corpus) - 10} more")
