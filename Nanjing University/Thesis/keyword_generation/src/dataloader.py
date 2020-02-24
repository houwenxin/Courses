# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:38:38 2019

@author: houwenxin

"""

import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.multiprocessing as multiprocessing
import numpy as np
import threading
import queue
import sys
import collections
import traceback
import re
import itertools
from torch.autograd import Variable

from vocab import PAD, BOS, EOS

string_classes = (str, bytes)

class KeywordDataset(torch.utils.data.Dataset):
    def __init__(self, data, word2id, id2word, type="one2many", include_original=False):
        if type == "one2many":
            keys = ["text_id", "keywords_id", "text_id_oov", "keywords_id_oov", "oov_dict", "oov_list"]
            if include_original:
                keys += ["text_tokens", "keywords_tokens"]
        elif type == "one2one":
            keys = ["text_id", "keyword_id", "text_id_oov", "keyword_id_oov", "oov_dict", "oov_list"]
            if include_original:
                keys += ["text_tokens", "keyword_tokens"]
        modified_data = []
        
        for single in data:
            modified_single = {}
            for key in keys:
                modified_single[key] = single[key]
            if "oov_list" in modified_single:
                if type == 'one2one':
                    modified_single['oov_number'] = len(modified_single['oov_list'])
                elif type == 'one2many':
                    modified_single['oov_number'] = [len(oov) for oov in modified_single['oov_list']]
            modified_data.append(modified_single)      
        self.data = modified_data
        self.word2id = word2id
        self.id2word = id2word
        self.pad_id = word2id[PAD]
        self.type = type
        self.include_original = include_original 
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
    def _pad(self, id_lists):
        id_lists = np.asarray(id_lists)
        id_lens = [len(id_list) for id_list in id_lists]
        max_len = max(id_lens)
        id_pad_lists = np.array([np.concatenate((id_list, [self.pad_id] * (max_len - len(id_list)))) for id_list in id_lists])
        id_pad_lists = Variable(torch.stack([torch.from_numpy(id_pad_list) for id_pad_list in id_pad_lists], 0)).type("torch.LongTensor")
        # mask将pad的部分标为0，非pad的部分标为1
        id_pad_masks = np.array([[1] * id_len + [0] * (max_len - id_len) for id_len in id_lens])
        id_pad_masks = Variable(torch.stack([torch.from_numpy(mask) for mask in id_pad_masks], 0))
        
        assert id_pad_lists.size(1) == max_len
        # 返回的id_pad_lists和id_pad_masks都是torch.autograd.Variable的格式，id_lens是一个二维list
        return id_pad_lists, id_lens, id_pad_masks
        
    def collate_fn_one2many(self, batches):
        # 源码中输入的batch: [self.dataset[i] for i in indices]
        # 一个batch就是一个文本及其对应的keywords
        text_id_lists = [[self.word2id[BOS]] + batch["text_id"] + [self.word2id[EOS]] for batch in batches]
        text_id_oov_lists = [[self.word2id[BOS]] + batch["text_id_oov"] + [self.word2id[EOS]] for batch in batches]
        keywords_id_lists = [[[self.word2id[BOS]] + keyword_id + [self.word2id[EOS]] for keyword_id in batch["keywords_id"]] for batch in batches]
        # 用于计算损失函数的目标关键词id
        keywords_id_target_lists = [[keyword_id + [self.word2id[EOS]] for keyword_id in batch['keywords_id']] for batch in batches]
        keywords_id_oov_target_lists = [[keyword_id_oov + [self.word2id[EOS]] for keyword_id_oov in batch["keywords_id_oov"]] for batch in batches]
        oov_lists = [batch["oov_list"] for batch in batches]
        
        if self.include_original:
            text_tokens_lists = [batch["text_tokens"] for batch in batches]
            keywords_tokens_lists = [batch["keywords_tokens"] for batch in batches]
        # 按text长度的倒序对这个batch中的所有数据重排顺序
        text_order_by_len = np.argsort([len(text_id_list) for text_id_list in text_id_lists])[::-1]
        text_id_lists = [text_id_lists[idx] for idx in text_order_by_len]
        text_id_oov_lists = [text_id_oov_lists[idx] for idx in text_order_by_len]
        keywords_id_lists = [keywords_id_lists[idx] for idx in text_order_by_len]
        keywords_id_target_lists = [keywords_id_target_lists[idx] for idx in text_order_by_len]
        keywords_id_oov_target_lists = [keywords_id_oov_target_lists[idx] for idx in text_order_by_len]
        oov_lists = [oov_lists[idx] for idx in text_order_by_len]
        if self.include_original:
            text_tokens_lists = [text_tokens_lists[idx] for idx in text_order_by_len]
            keywords_tokens_lists = [keywords_tokens_lists[idx] for idx in text_order_by_len]
        # 为了命名方便标为lists，其实这里pad返回的数据除了lens都是torch.autograd.Variable的格式。
        text_lists_one2many, text_one2many_lens, _ = self._pad(text_id_lists) # 格式：[[], [], []]
        keywords_lists_one2many = keywords_id_lists # 格式：[[[], [], []], [[], []], [[]]]
        text_oov_lists_one2many, _, _ = self._pad(text_id_oov_lists)
        keywords_oov_target_lists_one2many = keywords_id_oov_target_lists
        oov_lists_one2many = oov_lists
        
        # 接下来，将one2many的数据展开为one2one的数据
        text_lists_one2one, text_one2one_lens, _ = self._pad(list(itertools.chain(
                *[[text_id_lists[idx]] * len(keywords_num) for idx, keywords_num in enumerate(keywords_id_lists)])))
        text_oov_lists_one2one, _, _ = self._pad(list(itertools.chain(
                *[[text_id_oov_lists[idx]] * len(keywords_num) for idx, keywords_num in enumerate(keywords_id_lists)])))
        keywords_lists_one2one, _, _ = self._pad(list(itertools.chain(*[keywords_list for keywords_list in keywords_id_lists])))
        keywords_target_lists_one2one, _, _ = self._pad(list(itertools.chain(*[keywords_list for keywords_list in keywords_id_target_lists])))
        keywords_oov_target_lists_one2one, _, _ = self._pad(list(itertools.chain(*[keywords_list for keywords_list in keywords_id_oov_target_lists])))
        oov_lists_one2one = list(itertools.chain(
                *[[oov_lists[idx]] * len(keywords_num) for idx, keywords_num in enumerate(keywords_id_lists)]))
        
        #print(len(text_lists_one2one), len(text_oov_lists_one2one) \
        #        ,len(keywords_oov_target_lists_one2one), len(oov_lists_one2one))
        assert (len(text_id_lists) == len(text_lists_one2many) == len(text_oov_lists_one2many) \
                == len(keywords_oov_target_lists_one2many) == len(oov_lists_one2many))
        assert (sum([len(keyword_id_list) for keyword_id_list in keywords_id_lists]) \
                == len(text_lists_one2one) == len(text_oov_lists_one2one) \
                == len(keywords_oov_target_lists_one2one) == len(oov_lists_one2one))
        assert (text_lists_one2many.size() == text_oov_lists_one2many.size())
        assert (text_lists_one2one.size() == text_oov_lists_one2one.size())
        assert ([keywords_lists_one2one.size(0), keywords_lists_one2one.size(1) - 1] \
                 == list(keywords_target_lists_one2one.size()) == list(keywords_oov_target_lists_one2one.size()))
        
        if self.include_original:
            return (text_lists_one2many, text_one2many_lens, keywords_lists_one2many, None, 
                    keywords_oov_target_lists_one2many, text_oov_lists_one2many, oov_lists_one2many, 
                    text_tokens_lists, keywords_tokens_lists), (
                            text_lists_one2one, text_one2one_lens, keywords_lists_one2one, keywords_target_lists_one2one, 
                            keywords_oov_target_lists_one2one, text_oov_lists_one2one, oov_lists_one2one)
        else:
            return (text_lists_one2many, text_one2many_lens, keywords_lists_one2many, None, 
                    keywords_oov_target_lists_one2many, text_oov_lists_one2many, oov_lists_one2many), (
                            text_lists_one2one, text_one2one_lens, keywords_lists_one2one, keywords_target_lists_one2one, 
                            keywords_oov_target_lists_one2one, text_oov_lists_one2one, oov_lists_one2one)
    
"""
    numpy_type_map和default_collate 函数来自PyTorch源码
""" 
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}   
def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    raise TypeError((error_msg.format(type(batch[0]))))
class KeywordDataLoader(object):
    """
    关于DataLoader类的解读：https://blog.csdn.net/u012436149/article/details/78545766
    """
    def __init__(self, dataset, max_text_num=5, max_batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):
        self.dataset     = dataset
        # 用于产生one2many的batch
        self.num_keywords_list = [len(single["keywords_id"]) for single in dataset.data] # 每个example的关键词的个数，格式：[2, 3, 3, ...]
        self.batch_size         = max_batch_size # 因为要转化为one2one，所以这里的batch size其实就是batch中one2many样本的关键词个数的和
        self.max_text_num = max_text_num
        self.num_workers        = num_workers
        self.collate_fn         = collate_fn
        self.pin_memory         = pin_memory # If `True`, the data loader will copy tensors into CUDA pinned memory before returning them.
        self.drop_last          = drop_last # set to `True` to drop the last incomplete batch, 
                                            # if the dataset size is not divisible by the batch size. 
                                            # If `False` and the size of dataset is not divisible by the batch size, then the last batch
                                            # will be smaller. (default: False)
        if batch_sampler is not None:
            if self.batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')
        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')
        if self.num_workers < 0: 
            raise ValueError('num_workers cannot be negative; use num_workers=0 to disable multiprocessing.') 

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = One2ManyBatchSampler(sampler, 
                                                 self.num_keywords_list, 
                                                 max_text_num=self.max_text_num, 
                                                 max_batch_size=self.batch_size, 
                                                 drop_last=drop_last)
        self.sampler = sampler
        self.batch_sampler = batch_sampler
    def __iter__(self):
        return DataLoaderIter(self)
    def __len__(self):
        return len(self.batch_sampler)
    def one2one_number(self):
        return sum(self.num_keywords_list)
    
class One2ManyBatchSampler(object):
    def __init__(self, sampler, num_keywords_list, max_text_num, max_batch_size, drop_last):
        self.sampler = sampler
        self.num_keywords_list = num_keywords_list
        self.max_batch_size = max_batch_size # 因为要转化为one2one，所以这里的batch size其实就是batch中one2many样本的关键词个数的和
        self.max_text_num = max_text_num
        self.drop_last = drop_last
        batches = []
        batch = []
        for idx in self.sampler:
            # 当前batch中关键词的总个数
            num_keywords = sum([self.num_keywords_list[id] for id in batch])
            if len(batch) < self.max_text_num and num_keywords + self.num_keywords_list[idx] < self.max_batch_size:
                batch.append(idx)
            elif len(batch) == 0: # if the max_batch_size is very small, return a batch of only one data sample
                batch.append(idx)
                batches.append(batch)
                batch = []
            else: # 当batch装满之后，将其加入batches
                batches.append(batch)
                # print('batch %d: #(src)=%d, #(trg)=%d \t\t %s' % (len(batches), len(batch), number_trgs, str(batch)))
                batch = []
                batch.append(idx)
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        self.batches = batches
        self.num_batch = len(batches)
    def __iter__(self):
        return self.batches.__iter__()
    def __len__(self):
        return self.num_batch

"""
    以下代码到DataLoaderIter为止都来自PyTorch源码
"""
class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"
    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
def pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch
def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except Exception:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))
def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))  
class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()
            
def load_vocab_and_data(opts, logger, load_train=True):
    logger.info("Loading vocabulary from: %s" % opts.vocab)
    word2id, id2word, vocab = torch.load(opts.vocab, "rb")
    pin_memory = torch.cuda.is_available()
    if load_train:
        train_one2many = torch.load(opts.data_path + ".train.one2many.pth", "rb") # opt需要一个data_path的参数
        train_one2many_dataset = KeywordDataset(train_one2many, word2id=word2id, id2word=id2word, type='one2many')
        train_one2many_dataloader = KeywordDataLoader(dataset=train_one2many_dataset,
                                                    collate_fn=train_one2many_dataset.collate_fn_one2many,
                                                    num_workers=opts.batch_workers,
                                                    max_text_num=1024,
                                                    max_batch_size=opts.batch_size,
                                                    pin_memory=pin_memory,
                                                    shuffle=True)
        logger.info('Train data size: %d one2many pairs, %d one2one pairs, %d batches, average pair num/batch num: %.3f' 
                 % (len(train_one2many_dataloader.dataset), 
                    train_one2many_dataloader.one2one_number(), 
                    len(train_one2many_dataloader), 
                    train_one2many_dataloader.one2one_number() / len(train_one2many_dataloader)))
    else:
        train_one2many_dataloader = None
    
    valid_one2many = torch.load(opts.data_path + '.valid.one2many.pth', 'rb')
    test_one2many = torch.load(opts.data_path + '.test.one2many.pth', 'rb')

    # !important. As it takes too long to do beam search, thus reduce the size of validation and test datasets
    valid_one2many = valid_one2many[:2000]
    # test_one2many = test_one2many[:2000]

    valid_one2many_dataset = KeywordDataset(valid_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)
    test_one2many_dataset = KeywordDataset(test_one2many, word2id=word2id, id2word=id2word, type='one2many', include_original=True)

    valid_one2many_dataloader = KeywordDataLoader(dataset=valid_one2many_dataset,
                                                collate_fn=valid_one2many_dataset.collate_fn_one2many,
                                                num_workers=opts.batch_workers,
                                                max_text_num=opts.beam_search_batch_example,
                                                max_batch_size=opts.beam_search_batch_size,
                                                pin_memory=pin_memory,
                                                shuffle=False)
    test_one2many_dataloader = KeywordDataLoader(dataset=test_one2many_dataset,
                                               collate_fn=test_one2many_dataset.collate_fn_one2many,
                                               num_workers=opts.batch_workers,
                                               max_text_num=opts.beam_search_batch_example,
                                               max_batch_size=opts.beam_search_batch_size,
                                               pin_memory=pin_memory,
                                               shuffle=False)

    opts.word2id = word2id
    opts.id2word = id2word
    opts.vocab = vocab

    logger.info('Valid data size: %d one2many pairs, %d one2one pairs, %d batches' 
             % (len(valid_one2many_dataloader.dataset), valid_one2many_dataloader.one2one_number(), len(valid_one2many_dataloader)))
    logger.info('Test data size: %d one2many pairs, %d one2one pairs, %d batches' 
             % (len(test_one2many_dataloader.dataset), test_one2many_dataloader.one2one_number(), len(test_one2many_dataloader)))

    logger.info('Vocab size: %d' % len(vocab))
    logger.info('Used vocab size: %d' % opts.vocab_size)

    return train_one2many_dataloader, valid_one2many_dataloader, test_one2many_dataloader, word2id, id2word, vocab