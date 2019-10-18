import sentencepiece as spm
from tqdm import tqdm
import os

g_corpus_name = 'channel'
# g_corpus_name = 'hongloumeng'
g_text_file_with_sep_path = 'data-sp/{}/train_sep.txt'.format(g_corpus_name)
g_tokenized_data_dir_path = 'data-sp/{}/tokenized_sss'.format(g_corpus_name)
g_vocab_size = 16000
g_model_prefix = '{}_sp_model_{}'.format(g_corpus_name, g_vocab_size)
g_model_file = '{}/{}/{}.model'.format('cache-sp', g_corpus_name, g_model_prefix)
# g_num_pieces = 100
g_num_pieces = 10
# g_num_pieces = 1


def build_files(raw_data_path, tokenized_data_path, model_file, num_pieces):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    with open(raw_data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = f.readlines()
        single = ''.join(lines)
        len_single = len(single)

        sp = spm.SentencePieceProcessor()
        sp.load(model_file)
        for i in tqdm(range(num_pieces)):
            single_ids = sp.EncodeAsIds(single[len_single // num_pieces * i: len_single // num_pieces * (i + 1)])

            with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(i)), 'w') as ff:
                for token_id in single_ids[:-1]:
                    ff.write(str(token_id) + ' ')
                ff.write(str(single_ids[-1]))
                ff.write('\n')

    print('finish')


def test_take_a_look(tokenized_data_path, model_file, n):
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(n)), 'r') as f:
        line = f.read(1024).strip()
    tokens = line.split()
    tokens = [int(token) for token in tokens]
    text = sp.DecodeIds(tokens)
    print(text)


if __name__ == '__main__':
    # build_files(raw_data_path=g_text_file_with_sep_path, tokenized_data_path=g_tokenized_data_dir_path,
    #             model_file=g_model_file, num_pieces=g_num_pieces)
    test_take_a_look(tokenized_data_path=g_tokenized_data_dir_path, model_file=g_model_file, n=0)
