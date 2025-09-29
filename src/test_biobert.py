from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load model and tokenizer from checkpoint
tokenizer = AutoTokenizer.from_pretrained('results/checkpoint-200')
model = AutoModelForTokenClassification.from_pretrained('results/checkpoint-200')

# Define labels
label_list = ['O', 'B-ADE', 'I-ADE', 'B-DRUG', 'I-DRUG']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

# Test sentence for VAERS_ID 872934
test_text = "stroke Information has been received from a lawyer regarding a case in litigation and refers to a male patient (pt) of an unknown age. No information was provided regarding medical history, concurrent conditions, or concomitant medications. On or about 14-FEB-2014, the pt was inoculated with zoster vaccine live (ZOSTAVAX) as prescribed and/or administered by a pharmacist at a pharmacy (strength, dose, dose number, route, anatomical site of vaccination, lot number and expiration date were not provided) for the long-term prevention of shingles and/or zoster-related conditions. On an unknown date (reported as subsequent to the pt's zoster vaccine live (ZOSTAVAX) inoculation), the pt was hospitalized and treated by a healthcare provider for stroke (cerebrovascular accident). As a direct and proximate result of pt's use of the zoster vaccine live (ZOSTAVAX) vaccine, the pt had and would continued suffer ongoing injuries, including but not limited to: mental and physical pain and suffering; medical care and treatment for these injuries; significant medical and related expenses as a result of these injuries, including but not limited to medical losses and costs include care for hospitalization, physician care, monitoring, treatment, medications, and supplies; diminished capacity for the enjoyment of life; diminished quality of life; increased risk of premature death, aggravation of preexisting conditions and activation of latent conditions; and other losses and damages; and would continue to suffer such losses, and damages in the future. At the time of this report, the outcome of cerebrovascular accident was unknown. The reporter assessed that the event was related to zoster vaccine live (ZOSTAVAX). Upon internal review, cerebrovascular accident was determined to be a medically significant event."

# Tokenize with NLTK
tokens = word_tokenize(test_text.lower())
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
word_ids = inputs.word_ids()

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)[0].numpy()

# Align tokens and labels
filtered_tokens = []
filtered_labels = []
current_token = ""
current_label = None
previous_word_id = None
for i, (token, label, word_id) in enumerate(zip(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), predictions, word_ids)):
    if token in ['[CLS]', '[SEP]', '[PAD]']:
        continue
    if token in [',', '.', ':', ';', '(', ')', '/']:
        if current_token and current_label is not None:
            filtered_tokens.append(current_token)
            filtered_labels.append(id2label[current_label])
        filtered_tokens.append(token)
        filtered_labels.append('O')  # Force punctuation to O
        current_token = ""
        current_label = None
        continue
    if word_id is None:
        continue
    if word_id != previous_word_id and current_token and current_label is not None:
        filtered_tokens.append(current_token)
        filtered_labels.append(id2label[current_label])
        current_token = ""
        current_label = None
    if token.startswith('##'):
        current_token += token[2:]
    else:
        current_token += token
    current_label = label
    previous_word_id = word_id
if current_token and current_label is not None:
    filtered_tokens.append(current_token)
    filtered_labels.append(id2label[current_label])

# Load ground truth
iob_df = pd.read_csv('data/processed/sample_2020_gold_iob.csv')
ground_truth = iob_df[iob_df['VAERS_ID'] == 872934][['Token', 'Tag']].values.tolist()
ground_tokens = [str(row[0]).lower() for row in ground_truth if row[0]]
ground_labels = [row[1] for row in ground_truth if row[0]]

# Evaluate predictions
matched_labels = []
matched_ground = []
for pred_token, pred_label in zip(filtered_tokens, filtered_labels):
    for gt_token, gt_label in zip(ground_tokens, ground_labels):
        if pred_token.lower() == gt_token.lower():
            matched_labels.append(pred_label)
            matched_ground.append(gt_label)
            break

precision, recall, f1, _ = precision_recall_fscore_support(matched_ground, matched_labels, average='weighted', zero_division=0)

# Print predictions
print("Predictions:")
for token, label in zip(filtered_tokens, filtered_labels):
    print(f"{token}: {label}")

# Print evaluation metrics
print(f"\nEvaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save predictions
pd.DataFrame({"Token": filtered_tokens, "Label": filtered_labels}).to_csv('data/processed/test_predictions_checkpoint200.csv', index=False)
print("Saved predictions to data/processed/test_predictions_checkpoint200.csv")
