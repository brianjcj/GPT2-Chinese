# -*- coding: utf-8 -*-

import sentencepiece as spm
import os

g_corpus_name = 'channel'
# g_corpus_name = 'hongloumeng'
g_text_file_with_sep_path = 'data-sp/{}/train_sep.txt'.format(g_corpus_name)
g_vocab_size = 16000
g_return_token = 'Й'
g_model_prefix = 'cache-sp/{}/{}_sp_model_{}'.format(g_corpus_name, g_corpus_name, g_vocab_size)


def append_each_line_with_sep(text_file_path, text_file_with_seq_path, return_token):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        with open(text_file_with_seq_path, 'w', encoding='utf-8') as out_file:
            for line in f:
                out_file.write(line.strip() + return_token + '\n')


def main(text_file_with_seq_path, model_prefix, vocab_size):
    # train_args = '--input={} --model_prefix={} --vocab_size={}'.format(text_file_with_seq_path, model_prefix, vocab_size)
    train_args = '--input={} --model_prefix={} --vocab_size={} --character_coverage=1.0'.format(text_file_with_seq_path, model_prefix, vocab_size)
    print(train_args)
    spm.SentencePieceTrainer.train(train_args)


if __name__ == '__main__':
    dir_name = os.path.dirname(g_model_prefix)
    if dir_name != '' and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # append_each_line_with_sep(g_text_file_path, g_text_file_with_sep_path, g_return_token)
    main(text_file_with_seq_path=g_text_file_with_sep_path, model_prefix=g_model_prefix, vocab_size=g_vocab_size)


