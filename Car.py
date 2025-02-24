from tkinter import Tk, Label, Button, messagebox
from tkinter import ttk
from ttkthemes import ThemedTk  # For a themed Tkinter window

class CarPricePredictionApp:
    def __init__(self, root, car_data, model, scaler):
        self.root = root
        self.model = model
        self.scaler = scaler
        self.car_data = car_data

        # GUI Layout
        self.root.title("Car Price Prediction")
        self.root.geometry("600x400")
        self.root.configure(bg="#f7f7f7")  # Light background color

        # Style Configuration
        style = ttk.Style(self.root)
        style.theme_use('clam')  # Apply a modern theme
        style.configure('TLabel', font=('Arial', 12), background="#f7f7f7", foreground="#333333")
        style.configure('TButton', font=('Arial', 10, 'bold'), background="#4CAF50", foreground="white")
        style.map('TButton', background=[('active', '#45a049')])

        # Title Label
        Label(self.root, text="Car Price Prediction", font=("Arial", 20, "bold"), bg="#f7f7f7", fg="#333333").pack(pady=10)

        # Car Selector
        Label(self.root, text="Select Car Model:", font=("Arial", 12), bg="#f7f7f7").pack(pady=5)
        self.car_selector = ttk.Combobox(self.root, values=list(self.car_data['Car_Name'].unique()), state="readonly", font=("Arial", 10))
        self.car_selector.pack(pady=5)
        self.car_selector.set("Select a car model")

        # Predict Button
        Button(self.root, text="Predict Details", font=("Arial", 12), bg="#4CAF50", fg="white", relief="flat", command=self.predict_details).pack(pady=15)

        # Result Label
        self.result_frame = ttk.Frame(self.root, style='TFrame')
        self.result_frame.pack(pady=10, fill='x', padx=10)
        self.result_label = Label(self.result_frame, text="", font=("Arial", 12), justify="left", bg="#f7f7f7")
        self.result_label.pack(pady=5)

    def predict_details(self):
        selected_car = self.car_selector.get()
        if selected_car == "Select a car model":
            messagebox.showwarning("Selection Error", "Please select a car model.")
            return
        
        car_details = self.car_data[self.car_data['Car_Name'] == selected_car].iloc[0]
        input_features = car_details.drop(['Car_Name', 'Selling_Price']).values.reshape(1, -1)
        scaled_features = self.scaler.transform(input_features)
        
        predicted_price = self.model.predict(scaled_features)[0]
        year_of_sale = 2024  # Assuming predictions are for the current year
        current_price = car_details['Present_Price']
        distance_driven = car_details['Kms_Driven']
        
        self.result_label.config(text=(f"Details for {selected_car}:\n"
                                       f"  - Predicted Selling Price: ₹{predicted_price:.2f} Lakhs\n"
                                       f"  - Current Price: ₹{current_price:.2f} Lakhs\n"
                                       f"  - Distance Driven: {distance_driven} KM\n"
                                       f"  - Year of Prediction: {year_of_sale}"))

# Run GUI
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    import pandas as pd

    # Load and preprocess dataset
    car_data = pd.read_csv('car data.csv')
    car_data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    car_data = pd.get_dummies(car_data, columns=['Seller_Type', 'Transmission'], drop_first=True)
    X = car_data.drop(['Car_Name', 'Selling_Price'], axis=1)
    y = car_data['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Start the app
    root = ThemedTk(theme="arc")  # Use ThemedTk for better visuals
    app = CarPricePredictionApp(root, car_data, model, scaler)
    root.mainloop()
