# Grocery Purchase Proposal

A command-line tool that suggests one more grocery item based on what is already in the basket. It mines association rules with the Apriori algorithm (via `mlxtend`) from a grocery transaction dataset and answers with the consequent of the most confident matching rule.

## Data

`Groceries_dataset.csv` (included) is the public Groceries dataset from Kaggle. It has 38,765 rows with the columns `Member_number`, `Date` and `itemDescription`, one purchased item per row.

All items bought by the same member on the same date are treated as one basket. That gives 14,963 baskets over 167 distinct items.

## How it works

1. Group the rows into baskets, lowercase the item names and one-hot encode the baskets with `TransactionEncoder`.
2. Run Apriori with a minimum support of 0.5%. If nothing is found, retry with 0.2% and then 0.1%.
3. Generate rules with a minimum confidence of 1% (falling back to 0.5% and 0.1% if no rules come out).
4. For the items you type (1 to 3, comma separated):
   - look for rules whose antecedent is exactly that item set,
   - otherwise use rules for the first item only,
   - skip rules that would suggest an item you already entered,
   - print the consequent of the rule with the highest confidence.

With the default thresholds the script finds 126 frequent itemsets and 74 rules. The rules with the highest confidence all point to whole milk, which is the most common item in the data (2,502 rows):

```
bottled beer -> whole milk (confidence: 15.78%)
sausage -> whole milk (confidence: 14.84%)
newspapers -> whole milk (confidence: 14.43%)
domestic eggs -> whole milk (confidence: 14.23%)
frankfurter -> whole milk (confidence: 13.98%)
```

Confidence values are low overall because the baskets are small (a member/date pair has only a few items).

## Running

```bash
pip install -r requirements.txt
python purch_proposal.py
```

Example session:

```
Enter items: sausage, yogurt
Maybe you would also like to purchase whole milk

Enter items: whole milk
Maybe you would also like to purchase other vegetables

Enter items: q
```

Item names must match the dataset (case does not matter). The script prints the full item list at startup. Type `q`, `quit` or `exit` to stop.

## Files

```
purch_proposal.py       the tool
Groceries_dataset.csv   transaction data
requirements.txt
LICENSE
```

## License

MIT, see [LICENSE](LICENSE).
