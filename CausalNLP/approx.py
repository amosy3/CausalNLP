import torch


def get_logits_from_text(model, tokenizer, sentences):
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        # Get input_ids and attention mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        return outputs.logits

def get_cf_sample(concepts, current_concept, full_df, org_smaple, allways_return=False):
                    keep_concepts = [z for z in concepts if z != current_concept]
                    diff_concepts = [current_concept] #['Candidate_id', c]
                    matching_rows = full_df[(full_df[keep_concepts] == org_smaple[keep_concepts].values).all(axis=1)]
                    valid_cf = matching_rows[(matching_rows[diff_concepts] != org_smaple[diff_concepts].values).all(axis=1)]
                    if valid_cf.shape[0] == 0.0 and allways_return:
                        print('No counterfactuals for %s->%s !!!!' % (org_concept, do_concept))
                        return full_df.sample()
                    else:
                        return valid_cf.sample()