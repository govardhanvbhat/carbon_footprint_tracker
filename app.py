import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management

DATABASE = 'database.db'

# ---------------- Database Utility ----------------
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- Dashboard Plots ----------------
def plot_emission_trends(activities):
    if len(activities) == 0:
        return None
    df = pd.DataFrame(activities)
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date')['carbon_emission'].sum().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(df['date'], df['carbon_emission'], marker='o', color='blue')
    plt.title('Carbon Emission Trend')
    plt.xlabel('Date')
    plt.ylabel('CO2 Emission (kg)')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def plot_category_emissions(activities):
    if len(activities) == 0:
        return None
    df = pd.DataFrame(activities)
    cat_sum = df.groupby('activity_type')['carbon_emission'].sum()
    plt.figure(figsize=(8, 4))
    cat_sum.plot(kind='bar', color='green')
    plt.title('Emissions by Category')
    plt.ylabel('CO2 Emission (kg)')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def forecast_emission(activities):
    if len(activities) < 2:
        return None
    df = pd.DataFrame(activities)
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date')['carbon_emission'].sum().reset_index()
    df['days'] = (df['date'] - df['date'].min()).dt.days
    X = df[['days']]
    y = df['carbon_emission']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    future_days = np.arange(df['days'].max() + 1, df['days'].max() + 31).reshape(-1, 1)
    forecast = model.predict(future_days)
    plt.figure(figsize=(8, 4))
    plt.plot(df['date'], y, marker='o', color='blue', label='Actual')
    future_dates = pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=30)
    plt.plot(future_dates, forecast, marker='x', color='red', linestyle='--', label='Forecast')
    plt.title('Carbon Emission Forecast (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('CO2 Emission (kg)')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

# ---------------- Authentication ----------------
@app.route('/')
def welcome():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('welcome.html')

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)
app.secret_key = "your_secret_key"

def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn
def ai_generate_suggestions(activities):
    suggestions = []
    total_emission = 0

    # Calculate total emissions
    for activity in activities:
        emission = float(activity['carbon_emission'])
        total_emission += emission

        # AI-based analysis for each activity
        if emission < 50:
            suggestions.append(f"For activity '{activity['activity_name']}', carbon content is within a safe range.")
        elif 50 <= emission < 150:
            suggestions.append(f"'{activity['activity_name']}' has moderate carbon impact. Consider minor improvements.")
        else:
            suggestions.append(f"High carbon emission detected in '{activity['activity_name']}'. Try reducing usage or optimizing processes.")

    # Overall assessment
    if total_emission < 200:
        suggestions.append("✅ Overall emissions are well under control. Keep it up!")
    elif 200 <= total_emission < 500:
        suggestions.append("⚠️ Emissions are moderate. Look into optimizing transport, energy, or waste management.")
    else:
        suggestions.append("🚨 High overall carbon footprint detected. Immediate reduction actions recommended!")

    return suggestions


# ---------------- Home ----------------
# ==============================
# 🤖 Global AI Suggestion Function
# ==============================
def ai_generate_suggestions(activities, categories=None):
    suggestions = []
    total_emission = sum(a['emission'] for a in activities)

    # Individual activity feedback
    for act in activities:
        e = act['emission']
        name = f"{act['category']} - {act['subactivity']}"
        if e < 80:
            suggestions.append(f"✅ {name}: Carbon emission is minimal. Excellent sustainability choice!")
        elif 80 <= e < 200:
            suggestions.append(f"⚠️ {name}: Moderate emission. Consider optimizing frequency or efficiency.")
        else:
            suggestions.append(f"🚨 {name}: High emission! Explore eco-alternatives or reduce usage.")

    # Category-based advice
    if categories:
        for cat, val in categories.items():
            if val > 0:
                if val < 150:
                    suggestions.append(f"🌿 {cat}: Great job! Low overall emissions in this area.")
                elif 150 <= val < 400:
                    suggestions.append(f"🌍 {cat}: Moderate emissions — keep an eye on energy use and transport habits.")
                else:
                    suggestions.append(f"🔥 {cat}: High emissions detected! Consider major changes like renewables or reduced travel.")

    # Total summary
    if total_emission < 400:
        suggestions.append("💚 Overall: Excellent total footprint — keep maintaining your lifestyle!")
    elif 400 <= total_emission < 900:
        suggestions.append("🌱 Overall: Moderate footprint — small lifestyle changes can make a big difference.")
    else:
        suggestions.append("💥 Overall: High total carbon footprint! Consider major shifts like green transport and plant-based diet.")

    return suggestions


# ==============================
# 🏠 HOME ROUTE
# ==============================
@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db()
    conn.row_factory = sqlite3.Row
    activities = conn.execute(
        'SELECT activity_type, carbon_emission FROM activities WHERE user_id=?',
        (user_id,)
    ).fetchall()
    conn.close()

    if not activities:
        return render_template(
            'home.html',
            suggestions=["Add your first activity to get personalized AI insights."],
            plot_data=None
        )

    activities = [dict(a) for a in activities]

    # Parse and group activities by category
    categories = {'Transportation': 0, 'Energy': 0, 'Diet': 0, 'Household': 0, 'Other': 0}
    parsed_activities = []

    for act in activities:
        raw_type = act.get('activity_type', '').lower()
        parts = raw_type.split(',')
        category = parts[0].strip().capitalize() if len(parts) > 0 else "Other"
        subactivity = parts[1].strip() if len(parts) > 1 else "General"
        emission = float(act.get('carbon_emission', 0))
        categories[category] = categories.get(category, 0) + emission

        parsed_activities.append({
            'category': category,
            'subactivity': subactivity,
            'emission': emission
        })

    # Generate AI suggestions
    suggestions = ai_generate_suggestions(parsed_activities, categories)

    # Create pie chart
    labels = [k for k, v in categories.items() if v > 0]
    values = [v for v in categories.values() if v > 0]

    plot_data = None
    if values:
        plt.figure(figsize=(5, 5))
        plt.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=['#66bb6a', '#81c784', '#a5d6a7', '#388e3c', '#c8e6c9']
        )
        plt.title('Your Carbon Footprint Breakdown')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return render_template('home.html', suggestions=suggestions, plot_data=plot_data)


# ==============================
# ⚡ AJAX Endpoint for Live Updates
# ==============================
@app.route('/update_suggestions', methods=['POST'])
def update_suggestions():
    if 'user_id' not in session:
        return {'error': 'Unauthorized'}, 401

    user_id = session['user_id']
    data = request.get_json()
    activity_type = data.get('activity_type')
    carbon_emission = float(data.get('carbon_emission', 0))

    # Save new activity
    conn = get_db()
    conn.execute(
        'INSERT INTO activities (user_id, activity_type, carbon_emission) VALUES (?, ?, ?)',
        (user_id, activity_type, carbon_emission)
    )
    conn.commit()

    # Fetch all updated activities
    conn.row_factory = sqlite3.Row
    activities = conn.execute(
        'SELECT activity_type, carbon_emission FROM activities WHERE user_id=?',
        (user_id,)
    ).fetchall()
    conn.close()

    activities = [dict(a) for a in activities]

    # Rebuild data for AI
    categories = {'Transportation': 0, 'Energy': 0, 'Diet': 0, 'Household': 0, 'Other': 0}
    parsed_activities = []

    for act in activities:
        raw_type = act.get('activity_type', '').lower()
        parts = raw_type.split(',')
        category = parts[0].strip().capitalize() if len(parts) > 0 else "Other"
        subactivity = parts[1].strip() if len(parts) > 1 else "General"
        emission = float(act.get('carbon_emission', 0))
        categories[category] = categories.get(category, 0) + emission

        parsed_activities.append({
            'category': category,
            'subactivity': subactivity,
            'emission': emission
        })

    # Generate new suggestions
    suggestions = ai_generate_suggestions(parsed_activities, categories)

    # Generate pie chart
    labels = [k for k, v in categories.items() if v > 0]
    values = [v for v in categories.values() if v > 0]
    plot_data = None

    if values:
        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return {'suggestions': suggestions, 'plot_data': plot_data}


import re  # Add this import at the top

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        age = request.form.get('age')
        gender = request.form.get('gender')
        country = request.form.get('country')
        household_size = request.form.get('household_size')
        transport_mode = request.form.get('transport_mode')
        energy_source = request.form.get('energy_source')
        diet_type = request.form.get('diet_type')

        # ✅ Password strength validation
        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
            return redirect(url_for('register'))

        if not re.search(r"[A-Z]", password):
            flash("Password must contain at least one uppercase letter.", "danger")
            return redirect(url_for('register'))

        if not re.search(r"[a-z]", password):
            flash("Password must contain at least one lowercase letter.", "danger")
            return redirect(url_for('register'))

        if not re.search(r"[0-9]", password):
            flash("Password must contain at least one number.", "danger")
            return redirect(url_for('register'))

        if not re.search(r"[@$!%*?&]", password):
            flash("Password must contain at least one special character (@, $, !, %, *, ?, &).", "danger")
            return redirect(url_for('register'))

        # ✅ If valid, hash the password
        hashed_password = generate_password_hash(password)

        conn = get_db()
        cursor = conn.cursor()

        # ✅ Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            flash("Email already registered!", "danger")
            conn.close()
            return redirect(url_for('register'))

        # ✅ Insert new user
        cursor.execute("""
            INSERT INTO users (
                name, email, password, age, gender, country,
                household_size, transport_mode, energy_source, diet_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, email, hashed_password, age, gender, country,
            household_size, transport_mode, energy_source, diet_type
        ))
        conn.commit()
        conn.close()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for('home'))
        flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))

@app.route('/')
def welcome():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('welcome.html')


# ---------------- Dashboard ----------------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    user_id = session['user_id']
    activities = conn.execute('SELECT * FROM activities WHERE user_id=?', (user_id,)).fetchall()
    activities = [dict(row) for row in activities]
    conn.close()

    trend_plot = plot_emission_trends(activities)
    category_plot = plot_category_emissions(activities)
    forecast_plot = forecast_emission(activities)

    return render_template('dashboard.html',
                           trend_plot=trend_plot,
                           category_plot=category_plot,
                           forecast_plot=forecast_plot)

# ---------------- Add Activity ----------------
@app.route('/add_activity', methods=['GET', 'POST'])
def add_activity():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    activities_list = conn.execute('SELECT * FROM carbon_factors').fetchall()
    activities_list = [dict(a) for a in activities_list]

    if request.method == 'POST':
        user_id = session['user_id']
        activity_type = request.form['activity_type']
        activity_details = request.form['activity_details']
        quantity = float(request.form['quantity'])
        unit = request.form['unit']
        factor = conn.execute('SELECT emission_per_unit FROM carbon_factors WHERE subcategory=?',
                              (activity_type,)).fetchone()
        carbon_emission = quantity * factor['emission_per_unit'] if factor else 0
        conn.execute('INSERT INTO activities (user_id, activity_type, activity_details, quantity, unit, carbon_emission) VALUES (?,?,?,?,?,?)',
                     (user_id, activity_type, activity_details, quantity, unit, carbon_emission))
        conn.commit()
        conn.close()
        return redirect(url_for('dashboard'))

    conn.close()
    return render_template('add_activity.html', activities=activities_list)

# ---------------- Impact Simulator ----------------
# ---------------- Impact Simulator ----------------
@app.route('/impact_simulator', methods=['GET', 'POST'])
def impact_simulator():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    adjusted_plot = None
    user_id = session['user_id']

    conn = get_db()
    activities = conn.execute('SELECT * FROM activities WHERE user_id=?', (user_id,)).fetchall()
    activities = [dict(a) for a in activities]
    conn.close()

    # Only simulate if user already has some data
    if len(activities) == 0:
        flash("Please add some activities before running the simulator.", "info")
        return render_template('impact_simulator.html', adjusted_plot=None)

    # Base DataFrame
    df = pd.DataFrame(activities)

    if request.method == 'POST':
        transport_factor = float(request.form.get('transportation', 100)) / 100
        energy_factor = float(request.form.get('energy', 100)) / 100
        diet_factor = float(request.form.get('diet', 100)) / 100

        # Apply reduction to corresponding categories
        def adjust_emission(row):
            if 'transport' in row['activity_type'].lower():
                return row['carbon_emission'] * transport_factor
            elif 'energy' in row['activity_type'].lower():
                return row['carbon_emission'] * energy_factor
            else:
                return row['carbon_emission'] * diet_factor

        df['adjusted_emission'] = df.apply(adjust_emission, axis=1)

        # Compare original vs adjusted per category
        orig = df.groupby('activity_type')['carbon_emission'].sum()
        adjusted = df.groupby('activity_type')['adjusted_emission'].sum()

        # Plot comparison side-by-side
        plt.figure(figsize=(8, 4))
        index = np.arange(len(orig.index))
        bar_width = 0.35
        plt.bar(index, orig, bar_width, label='Original', color='red')
        plt.bar(index + bar_width, adjusted, bar_width, label='Adjusted', color='green')
        plt.xticks(index + bar_width / 2, orig.index, rotation=30)
        plt.title('Impact Simulation — Original vs Adjusted Emissions')
        plt.ylabel('CO2 Emission (kg)')
        plt.legend()
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        adjusted_plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return render_template('impact_simulator.html', adjusted_plot=adjusted_plot)
# ---------------- AJAX endpoint for live updates ----------------
# ---------------- AJAX endpoint for live updates ----------------
@app.route('/simulate_impact', methods=['POST'])
def simulate_impact():
    if 'user_id' not in session:
        return {'error': 'Not logged in'}

    user_id = session['user_id']
    data = request.get_json()

    # Extract slider adjustments (as percentages)
    transport_factor = float(data['adjustments'].get('transportation', 100)) / 100
    energy_factor = float(data['adjustments'].get('energy', 100)) / 100
    diet_factor = float(data['adjustments'].get('diet', 100)) / 100
    household_factor = float(data['adjustments'].get('household', 100)) / 100

    conn = get_db()
    activities = conn.execute('SELECT * FROM activities WHERE user_id=?', (user_id,)).fetchall()
    activities = [dict(a) for a in activities]
    conn.close()

    if len(activities) == 0:
        return {'error': 'No activities found'}

    df = pd.DataFrame(activities)

    # ✅ Helper to adjust emissions by keyword
    def adjust_emission(row):
        name = row['activity_type'].lower()
        name = name.replace('-', ' ').replace('_', ' ').replace('/', ' ')

        transport_keywords = [
            'transport', 'car', 'bike', 'bus', 'train', 'plane', 'flight', 'vehicle', 'commute', 'travel'
        ]
        energy_keywords = [
            'energy', 'electricity', 'power', 'heater', 'ac', 'appliance',
            'natural gas', 'gas', 'fuel', 'petrol', 'diesel', 'coal'
        ]
        diet_keywords = [
            'diet', 'food', 'meal', 'meat', 'fish', 'pork', 'beef', 'chicken',
            'lamb', 'rice', 'vegetable', 'fruit', 'grain', 'milk', 'dairy', 'fibers'
        ]
        household_keywords = [
            'household', 'home', 'laundry', 'cooking', 'kitchen',
            'cleaning', 'domestic', 'waste', 'recycling', 'water heating',
            'washing machine', 'dishwasher', 'air conditioning', 'heating'
        ]

        if any(k in name for k in transport_keywords):
            factor = transport_factor
        elif any(k in name for k in energy_keywords):
            factor = energy_factor
        elif any(k in name for k in diet_keywords):
            factor = diet_factor
        elif any(k in name for k in household_keywords):
            factor = household_factor
        else:
            factor = 1.0

        new_value = row['carbon_emission'] * factor

        # Optional: print for debug
        print(f"Activity: {row['activity_type']} | Adjusted: {new_value:.2f}")
        return new_value

    df['adjusted_emission'] = df.apply(adjust_emission, axis=1)

    # Calculate totals
    original_total = df['carbon_emission'].sum()
    adjusted_total = df['adjusted_emission'].sum()

    # Plot comparison
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Original', 'Adjusted'], [original_total, adjusted_total],
                   color=['#d32f2f', '#388e3c'])
    plt.title('Impact Simulation — Original vs Adjusted Emissions')
    plt.ylabel('CO₂ Emission (kg)')
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}', ha='center')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "original_total": round(original_total, 2),
        "adjusted_total": round(adjusted_total, 2),
        "adjusted_plot": plot_data
    }

if __name__ == '__main__':
    app.run(debug=True)
