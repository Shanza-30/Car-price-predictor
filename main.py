from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
lr_model = pickle.load(open("lr_car_model.pkl", "rb"))
rf_model = pickle.load(open("rf_car_model.pkl", "rb"))

# Encoding dictionaries for categorical features
company_encoding = {'Maruti':0, 'Hyundai':1, 'Honda':2, 'Toyota':3}
fuel_encoding = {'Petrol':0, 'Diesel':1, 'CNG':2}
transmission_encoding = {'Manual':0, 'Automatic':1}
owner_encoding = {'First':0, 'Second':1, 'Third':2}
drivetrain_encoding = {'FWD':0, 'RWD':1, 'AWD':2}
body_type_encoding = {'Sedan':0, 'Hatchback':1, 'SUV':2}
color_encoding = {'Red':0, 'Blue':1, 'Black':2, 'White':3}
seats_material_encoding = {'Fabric':0, 'Leather':1}

@app.route('/')
def home():
    return render_template('index.html',
                           companies=list(company_encoding.keys()),
                           fuels=list(fuel_encoding.keys()),
                           transmissions=list(transmission_encoding.keys()),
                           owners=list(owner_encoding.keys()),
                           drivetrains=list(drivetrain_encoding.keys()),
                           body_types=list(body_type_encoding.keys()),
                           colors=list(color_encoding.keys()),
                           seat_materials=list(seats_material_encoding.keys())
                           )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Encode categorical inputs
        company_val = company_encoding[request.form['Company']]
        fuel_val = fuel_encoding[request.form['Fuel_Type']]
        transmission_val = transmission_encoding[request.form['Transmission']]
        owner_val = owner_encoding[request.form['Owner_Type']]
        drivetrain_val = drivetrain_encoding[request.form['Drivetrain']]
        body_type_val = body_type_encoding[request.form['Body_Type']]
        color_val = color_encoding[request.form['Color']]
        seats_material_val = seats_material_encoding[request.form['Seats_Material']]

        # Numeric inputs
        year = float(request.form['Year'])
        mileage = float(request.form['Mileage'])
        engine = float(request.form['Engine'])
        max_power = float(request.form['Max_Power'])
        seats = float(request.form['Seats'])
        length = float(request.form['Length'])
        width = float(request.form['Width'])
        height = float(request.form['Height'])
        wheelbase = float(request.form['Wheelbase'])
        fuel_capacity = float(request.form['Fuel_Capacity'])
        boot_space = float(request.form['Boot_Space'])
        ground_clearance = float(request.form['Ground_Clearance'])
        curb_weight = float(request.form['Curb_Weight'])
        gross_weight = float(request.form['Gross_Weight'])
        top_speed = float(request.form['Top_Speed'])
        acceleration = float(request.form['Acceleration'])
        torque = float(request.form['Torque'])
        fuel_city = float(request.form['Fuel_Economy_City'])
        fuel_highway = float(request.form['Fuel_Economy_Highway'])
        co2_emission = float(request.form['CO2_Emissions'])
        safety_rating = float(request.form['Safety_Rating'])
        airbags = float(request.form['Airbags'])
        abs_val = float(request.form['ABS'])
        sunroof_val = float(request.form['Sunroof'])

        # Combine all features in training order (33 features)
        input_values = [
            company_val, 0, year, fuel_val, transmission_val, owner_val,
            mileage, engine, max_power, seats, length, width, height,
            wheelbase, fuel_capacity, boot_space, ground_clearance, curb_weight,
            gross_weight, top_speed, acceleration, torque, drivetrain_val, body_type_val,
            color_val, fuel_city, fuel_highway, co2_emission,
            seats_material_val, safety_rating, airbags, abs_val, sunroof_val
        ]

        input_array = np.array([input_values])  # shape (1,33)

        # Predictions
        lr_pred = lr_model.predict(input_array)[0]
        rf_pred = rf_model.predict(input_array)[0]

    except Exception as e:
        return f"Invalid input: {e}"

    return render_template('index.html',
                           companies=list(company_encoding.keys()),
                           fuels=list(fuel_encoding.keys()),
                           transmissions=list(transmission_encoding.keys()),
                           owners=list(owner_encoding.keys()),
                           drivetrains=list(drivetrain_encoding.keys()),
                           body_types=list(body_type_encoding.keys()),
                           colors=list(color_encoding.keys()),
                           seat_materials=list(seats_material_encoding.keys()),
                           linear_result=f"{lr_pred:.2f}",
                           rf_result=f"{rf_pred:.2f}"
                           )

if __name__ == "__main__":
    app.run(debug=True)
