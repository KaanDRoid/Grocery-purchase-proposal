import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuration constants
MIN_SUPPORT_PRIMARY = 0.005      # 0.5% - primary support threshold
MIN_SUPPORT_SECONDARY = 0.002    # 0.2% - fallback support threshold  
MIN_SUPPORT_MINIMUM = 0.001      # 0.1% - minimum support threshold
MIN_CONFIDENCE_PRIMARY = 0.01    # 1% - primary confidence threshold 
MIN_CONFIDENCE_SECONDARY = 0.005 # 0.5% - fallback confidence threshold
MIN_CONFIDENCE_MINIMUM = 0.001   # 0.1% - minimum confidence threshold

def load_groceries_data():
    """
    Load and prepare the groceries dataset for association rules mining.
    
    Returns:
        pd.DataFrame: Binary encoded transaction data
    """
    try:
        # Load the real groceries dataset
        print("Loading Groceries_dataset.csv...")
        data = pd.read_csv('Groceries_dataset.csv')
        
        # Group transactions by Member_number and Date to create market baskets
        transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()
        
        print(f"Loaded {len(transactions)} transactions from {len(data)} individual items")
        
        # Clean and standardize item names
        cleaned_transactions = []
        for transaction in transactions:
            # Convert to lowercase and strip whitespace
            cleaned_items = [item.strip().lower() for item in transaction if pd.notna(item)]
            if len(cleaned_items) > 0:  # Only include non-empty transactions
                cleaned_transactions.append(cleaned_items)
        
        print(f"After cleaning: {len(cleaned_transactions)} non-empty transactions")
        
        # Convert to binary encoded format using TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(cleaned_transactions).transform(cleaned_transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df
        
    except FileNotFoundError:
        print("Groceries_dataset.csv not found. Please ensure the file is in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_association_rules(df, min_confidence=MIN_CONFIDENCE_PRIMARY):
    """
    Generate association rules from the transaction data.
    
    Args:
        df (pd.DataFrame): Binary encoded transaction data
        min_confidence (float): Minimum confidence threshold (default: 1%)
    
    Returns:
        pd.DataFrame: Association rules dataframe
    """
    try:
        # Generate frequent itemsets using Apriori algorithm
        print("Generating frequent itemsets...")
        frequent_itemsets = apriori(df, min_support=MIN_SUPPORT_PRIMARY, use_colnames=True)
        
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Trying with lower support...")
            frequent_itemsets = apriori(df, min_support=MIN_SUPPORT_SECONDARY, use_colnames=True)
        
        if frequent_itemsets.empty:
            print("Still no frequent itemsets. Trying minimum support...")
            frequent_itemsets = apriori(df, min_support=MIN_SUPPORT_MINIMUM, use_colnames=True)
        
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, 
                                metric="confidence", 
                                min_threshold=min_confidence)
        
        if rules.empty:
            print("No rules found with current confidence. Trying with lower confidence...")
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence", 
                                    min_threshold=MIN_CONFIDENCE_SECONDARY)
        
        if rules.empty:
            print("Still no rules. Trying minimum confidence...")
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence", 
                                    min_threshold=MIN_CONFIDENCE_MINIMUM)
        
        return rules
        
    except Exception as e:
        print(f"Error generating association rules: {e}")
        return pd.DataFrame()

def parse_input_items(input_str):
    """
    Parse comma separated input items and clean them.
    
    Args:
        input_str (str): Comma separated item list
    
    Returns:
        list: Clean list of items (1-3 elements max)
    """
    items = [item.strip().lower() for item in input_str.split(',')]
    # Remove empty strings
    items = [item for item in items if item]
    
    # Limit to maximum 3 items
    if len(items) > 3:
        print(f"Warning: Only first 3 items will be considered: {items[:3]}")
        items = items[:3]
    
    return items

def find_best_recommendation(input_items, rules):
    """
    Find the best recommendation based on input items and association rules.
    
    Args:
        input_items (list): List of input items
        rules (pd.DataFrame): Association rules dataframe
    
    Returns:
        str: Recommended item or None if no recommendation found
    """
    if rules.empty:
        return None
    
    # Convert input items to frozenset for comparison
    input_set = frozenset(input_items)
    
    # First, try to find rules where the antecedent matches exactly
    matching_rules = rules[rules['antecedents'] == input_set]
    
    if not matching_rules.empty:
        # Return the consequent with highest confidence
        best_rule = matching_rules.loc[matching_rules['confidence'].idxmax()]
        consequent = list(best_rule['consequents'])[0]
        return consequent
    
    # If no exact match, try to find rules for the first item with highest confidence
    if input_items:
        first_item = input_items[0]
        first_item_set = frozenset([first_item])
        
        first_item_rules = rules[rules['antecedents'] == first_item_set]
        
        if not first_item_rules.empty:
            best_rule = first_item_rules.loc[first_item_rules['confidence'].idxmax()]
            consequent = list(best_rule['consequents'])[0]
            return consequent
    
    return None

def main():
    """
    Main function to run the purchase proposal system. I hope its look cool
    """
    print("=" * 60)
    print("  GROCERY PURCHASE PROPOSAL SYSTEM")
    print("=" * 60)
    print()
    
    # Load groceries dataset
    print("Loading groceries dataset...")
    df = load_groceries_data()
    
    if df is None:
        print("Failed to load dataset. Exiting...")
        return
    
    print(f"Dataset loaded successfully! {len(df)} transactions, {len(df.columns)} unique items")
    print(f"Available items: {', '.join(sorted(df.columns))}")
    print()
    
    # Generate association rules with minimum confidence of 1%
    print("Generating association rules (minimum confidence: 1%)...")
    rules = generate_association_rules(df, min_confidence=MIN_CONFIDENCE_PRIMARY)
    
    if rules.empty:
        print("No association rules generated. Please check your dataset.")
        return
    
    print(f"Generated {len(rules)} association rules")
    print()
    
    # Display top 5 rules for reference
    print("Top 5 Association Rules (by confidence):")
    print("-" * 50)
    top_rules = rules.nlargest(5, 'confidence')
    for idx, rule in top_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))
        con = ', '.join(list(rule['consequents']))
        conf = rule['confidence']
        print(f"  {ant} → {con} (confidence: {conf:.2%})")
    print()
    
    # Interactive recommendation loop 
    print("RECOMMENDATION SYSTEM")
    print("-" * 30)
    print("Enter items (comma-separated, 1-3 items max)")
    print("Example: whole milk, bread")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("Enter items: ").strip()
            
            # Check for exit condition
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the Purchase Proposal System! See ya!")
                break
            
            # Parse input items
            input_items = parse_input_items(user_input)
            
            if not input_items:
                print("Please enter at least one item.")
                continue
            
            # Validate items exist in dataset
            available_items = set(df.columns)
            valid_items = [item for item in input_items if item in available_items]
            
            if not valid_items:
                print(f"None of the items found in dataset.")
                print(f"Available items include: {', '.join(sorted(list(available_items)[:10]))}...")
                continue
            
            # Only warn if there were invalid items
            if len(valid_items) != len(input_items):
                invalid_items = [item for item in input_items if item not in available_items]
                print(f"Note: Items not found in dataset: {', '.join(invalid_items)}")
                input_items = valid_items
            
            # Find recommendation
            recommendation = find_best_recommendation(input_items, rules)
            
            if recommendation:
                print(f"Maybe you would also like to purchase {recommendation}")
            else:
                print("No recommendation available for the given item set.")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
