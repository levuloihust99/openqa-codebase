import tensorflow as tf


class Tensorizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tensorize_single(self, sequence, max_length):
        token_ids = self.tokenizer.encode(sequence)
        num_paddings = max(0, max_length - len(token_ids))
        token_ids += [self.tokenizer.pad_token_id] * num_paddings # padding
        token_ids = token_ids[:max_length] # truncating
        token_ids[-1] = self.tokenizer.sep_token_id # add '[SEP]' token add the end

        return tf.convert_to_tensor(token_ids)

    def tensorize_single_nonpad(self, sequence):
        return tf.convert_to_tensor(self.tokenizer.encode(sequence))

    def tensorize(self, sequences, max_length):
        tensors = []
        for sequence in sequences:
            token_ids = self.tensorize_single(sequence, max_length=max_length)
            tensors.append(token_ids)

        return tf.convert_to_tensor(tensors)

    def tensorize_question(self, questions, max_query_length):
        return self.tensorize(sequences=questions, max_length=max_query_length)

    def tensorize_question_nonpad(self, questions):
        if len(questions) == 0:
            return None
        return tf.ragged.constant([self.tokenizer.encode(q) for q in questions], dtype=tf.int32)

    def tensorize_context(self, contexts, max_context_length):
        tensors = []

        for context in contexts:
            title = context['title']
            text  = context['text']
            
            tokenized_title = self.tokenizer.tokenize(title)
            tokenized_text  = self.tokenizer.tokenize(text)

            tokens = [self.tokenizer.cls_token] + tokenized_title + [self.tokenizer.sep_token] \
                + tokenized_text + [self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            num_paddings = max(0, max_context_length - len(token_ids))
            token_ids += [self.tokenizer.pad_token_id] * num_paddings
            token_ids = token_ids[:max_context_length]
            token_ids[-1] = self.tokenizer.sep_token_id

            tensors.append(token_ids)            

        return tf.convert_to_tensor(tensors, dtype=tf.int32)

    def tensorize_context_nonpad(self, contexts):
        tensors = []

        for context in contexts:
            title = context['title']
            text = context['text']

            tokenized_title = self.tokenizer.tokenize(title)
            tokenized_text = self.tokenizer.tokenize(text)

            tokens = [self.tokenizer.cls_token] + tokenized_title + [self.tokenizer.sep_token] \
                + tokenized_text + [self.tokenizer.sep_token]

            tensors.append(self.tokenizer.convert_tokens_to_ids(tokens))
        
        if not tensors:
            return None
        return tf.ragged.constant(tensors, dtype=tf.int32)

