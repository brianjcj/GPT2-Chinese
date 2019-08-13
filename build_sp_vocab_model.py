# -*- coding: utf-8 -*-

import sentencepiece as spm

g_corpus_name = 'channel'
g_text_file_path = 'data-sp/{}/train.json'.format(g_corpus_name)
g_text_file_with_sep_path = 'data-sp/{}/train_sep.json'.format(g_corpus_name)
g_model_prefix = '{}_sp_model'.format(g_corpus_name)
g_vocab_size = 32000
g_return_token = 'Й'


def append_each_line_with_sep(text_file_path, text_file_with_seq_path, return_token):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        with open(text_file_with_seq_path, 'w', encoding='utf-8') as out_file:
            for line in f:
                out_file.write(line.strip() + return_token + '\n')


def main(text_file_with_seq_path, model_prefix, vocab_size):
    train_args = '--input={} --model_prefix={} --vocab_size={}'.format(text_file_with_seq_path, model_prefix, vocab_size)
    print(train_args)
    spm.SentencePieceTrainer.train(train_args)


if __name__ == '__main__':
    append_each_line_with_sep(g_text_file_path, g_text_file_with_sep_path, g_return_token)
    main(text_file_with_seq_path=g_text_file_with_sep_path, model_prefix=g_model_prefix, vocab_size=g_vocab_size)


