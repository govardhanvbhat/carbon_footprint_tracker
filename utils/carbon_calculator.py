def calculate_carbon(category, subcategory, quantity, conn):
    """
    Calculate carbon emissions based on category, subcategory, and quantity
    """
    cursor = conn.execute(
        'SELECT emission_per_unit, unit FROM carbon_factors WHERE category=? AND subcategory=?',
        (category, subcategory)
    )
    result = cursor.fetchone()
    if result:
        emission_factor, unit = result
        try:
            qty = float(quantity)
        except ValueError:
            qty = 1
        return emission_factor * qty
    else:
        return 0.0
