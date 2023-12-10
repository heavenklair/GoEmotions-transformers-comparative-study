import pandas as pd
import numpy as np
import re
import contractions

class TextCleaner:
    def __init__(self, df_train=None, df_val=None, df_test=None):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

    def lowercase_text(self):
        token_pattern = re.compile(r'\[[A-Z]+\]')

        def preserve_uppercase_tokens(text):
            parts = token_pattern.split(text) # Split the text into tokens
            tokens = token_pattern.findall(text) # Find all uppercase tokens
            lowered_parts = [part.lower() for part in parts] # Lowercase the non-token parts and keep tokens as they are
            result = ''.join(sum(zip(lowered_parts, tokens + ['']), ())) # Reconstruct the text
            return result

        for df in [self.df_train, self.df_val, self.df_test]:
            df['text'] = df['text'].apply(preserve_uppercase_tokens)

    def remove_urls_and_html(self, text):
        text = re.sub(r'http\\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        return text

    def apply_remove_urls_and_html(self):
        for df in [self.df_train, self.df_val, self.df_test]:
            df['text'] = df['text'].apply(self.remove_urls_and_html)

    def remove_reddit_prefix(self):
        pattern = r'^r/'
        for df in [self.df_train, self.df_val, self.df_test]:
            df.loc[df['text'].str.startswith('r/'), 'text'] = df['text'].str.replace(pattern, '', regex=True)

    def expand_contractions(self, text):
        return ' '.join([contractions.fix(word) for word in text.split()])

    def apply_expand_contractions(self):
        for df in [self.df_train, self.df_val, self.df_test]:
            df['text'] = df['text'].apply(self.expand_contractions)

    def remove_punctuation(self, text):
        punctuations_to_remove = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        token_pattern = re.compile(r'(\[[^\]]*\])')  # Pattern to capture tokens

        def remove_punct(text_part):
            return re.sub(rf'[{re.escape(punctuations_to_remove)}]', '', text_part)

        # Split text into tokens and non-tokens
        parts = token_pattern.split(text)
        # Process non-token parts to remove punctuation
        processed_parts = [remove_punct(part) if not token_pattern.match(part) else part for part in parts]
        # Reassemble the text
        return ''.join(processed_parts)



    
    def apply_remove_punctuation(self):
        for df in [self.df_train, self.df_val, self.df_test]:
            df['text'] = df['text'].apply(self.remove_punctuation)

    def remove_space(self, text):
        text = text.strip()
        return ' '.join(text.split())

    def apply_remove_space(self):
        for df in [self.df_train, self.df_val, self.df_test]:
            df['text'] = df['text'].apply(self.remove_space)

    def clean_all(self):
        self.lowercase_text()
        self.apply_remove_urls_and_html()
        self.remove_reddit_prefix()
        self.apply_expand_contractions()
        self.apply_remove_punctuation()
        self.apply_remove_space()

