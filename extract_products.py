#!/usr/bin/env python3
"""
Extract Product Recommendations from Care Plans
This script parses all existing care plans and extracts product recommendations
into a structured database table for tracking and management.
"""

import re
import psycopg2
import os
from urllib.parse import urlparse, parse_qs
import json

def connect_to_db():
    """Connect to PostgreSQL database using environment variables"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('PGHOST'),
            database=os.getenv('PGDATABASE'),
            user=os.getenv('PGUSER'),
            password=os.getenv('PGPASSWORD'),
            port=os.getenv('PGPORT')
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def extract_products_from_html(care_plan_html, case_id):
    """Extract product recommendations from HTML care plan content"""
    products = []
    
    # Pattern to match product recommendation blocks
    product_pattern = r'<div[^>]*background-color:#f9fafb[^>]*>.*?<h4[^>]*>([^<]+)</h4>.*?<p[^>]*>([^<]+)</p>.*?<a href="([^"]+)"[^>]*>.*?</div>'
    
    matches = re.findall(product_pattern, care_plan_html, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        name = match[0].strip()
        description = match[1].strip()
        amazon_url = match[2].strip()
        
        # Extract search keywords from Amazon URL
        keywords = extract_keywords_from_amazon_url(amazon_url)
        
        # Determine category based on product name and description
        category = categorize_product(name, description)
        
        # Determine wound types and audiences (we'll infer from context)
        wound_types = infer_wound_types(name, description)
        audiences = infer_audiences(description)
        
        products.append({
            'name': name,
            'category': category,
            'description': description,
            'amazon_search_url': amazon_url,
            'search_keywords': keywords,
            'wound_types': wound_types,
            'audiences': audiences,
            'extracted_from_case_id': case_id
        })
    
    return products

def extract_keywords_from_amazon_url(url):
    """Extract search keywords from Amazon URL"""
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        keywords = query_params.get('k', [''])[0]
        return keywords.replace('+', ' ')
    except:
        return ""

def categorize_product(name, description):
    """Categorize product based on name and description"""
    name_lower = name.lower()
    desc_lower = description.lower()
    
    if any(word in name_lower for word in ['dressing', 'bandage', 'gauze', 'pad']):
        return 'wound_dressing'
    elif any(word in name_lower for word in ['cleanser', 'cleansing', 'wash', 'saline']):
        return 'cleansing'
    elif any(word in name_lower for word in ['moisturizer', 'cream', 'lotion', 'barrier']):
        return 'moisturizing'
    elif any(word in name_lower for word in ['compression', 'stocking', 'sock']):
        return 'compression'
    elif any(word in name_lower for word in ['pillow', 'elevation', 'support']):
        return 'positioning'
    elif any(word in name_lower for word in ['antibiotic', 'antiseptic', 'ointment']):
        return 'medication'
    elif any(word in name_lower for word in ['tape', 'adhesive', 'securing']):
        return 'securing'
    else:
        return 'general'

def infer_wound_types(name, description):
    """Infer applicable wound types from product context"""
    wound_types = []
    combined = (name + " " + description).lower()
    
    if any(word in combined for word in ['venous', 'vein', 'varicose']):
        wound_types.append('venous_ulcer')
    if any(word in combined for word in ['diabetic', 'diabetes']):
        wound_types.append('diabetic_ulcer')
    if any(word in combined for word in ['pressure', 'bed sore', 'decubitus']):
        wound_types.append('pressure_ulcer')
    if any(word in combined for word in ['surgical', 'post-op', 'incision']):
        wound_types.append('surgical_wound')
    
    # If no specific wound type, assume general applicability
    if not wound_types:
        wound_types = ['general_wound_care']
    
    return wound_types

def infer_audiences(description):
    """Infer target audiences from description language"""
    desc_lower = description.lower()
    audiences = []
    
    # Simple language indicates family/patient audience
    if any(word in desc_lower for word in ['gentle', 'mild', 'easy', 'simple', 'comfortable']):
        audiences.extend(['family', 'patient'])
    
    # Medical terminology indicates professional audience
    if any(word in desc_lower for word in ['medical-grade', 'clinical', 'prescription', 'therapeutic']):
        audiences.append('professional')
    
    # Default to all audiences if unclear
    if not audiences:
        audiences = ['family', 'patient', 'professional']
    
    return audiences

def save_products_to_db(conn, products):
    """Save extracted products to database"""
    cursor = conn.cursor()
    
    for product in products:
        # Check if product already exists (by name)
        cursor.execute(
            "SELECT id FROM product_recommendations WHERE name = %s",
            (product['name'],)
        )
        
        if cursor.fetchone():
            print(f"Product '{product['name']}' already exists, skipping...")
            continue
        
        # Insert new product
        cursor.execute("""
            INSERT INTO product_recommendations 
            (name, category, description, amazon_search_url, search_keywords, 
             wound_types, audiences, extracted_from_case_id, priority, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            product['name'],
            product['category'],
            product['description'],
            product['amazon_search_url'],
            product['search_keywords'],
            product['wound_types'],
            product['audiences'],
            product['extracted_from_case_id'],
            50,  # Default priority
            True  # Active by default
        ))
        
        print(f"Added product: {product['name']} (Category: {product['category']})")
    
    conn.commit()

def main():
    """Main extraction process"""
    print("🔍 Extracting product recommendations from care plans...")
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Get all care plans
    cursor.execute("SELECT case_id, care_plan FROM wound_assessments WHERE care_plan IS NOT NULL")
    care_plans = cursor.fetchall()
    
    print(f"Found {len(care_plans)} care plans to process...")
    
    all_products = []
    
    for case_id, care_plan in care_plans:
        if care_plan:
            products = extract_products_from_html(care_plan, case_id)
            all_products.extend(products)
            print(f"Extracted {len(products)} products from case {case_id}")
    
    # Remove duplicates by name
    unique_products = {}
    for product in all_products:
        if product['name'] not in unique_products:
            unique_products[product['name']] = product
    
    unique_products_list = list(unique_products.values())
    print(f"Found {len(unique_products_list)} unique products total")
    
    # Save to database
    save_products_to_db(conn, unique_products_list)
    
    # Print summary
    cursor.execute("SELECT category, COUNT(*) FROM product_recommendations GROUP BY category")
    summary = cursor.fetchall()
    
    print("\n📊 Product Recommendations Summary:")
    for category, count in summary:
        print(f"  {category}: {count} products")
    
    cursor.close()
    conn.close()
    print("\n✅ Product extraction completed!")

if __name__ == "__main__":
    main()