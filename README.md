# AlphaCare Insurance Company Risk Analytics & Predictive Modeling
End-to-End Insurance Risk Analytics &amp; Predictive Modeling for  AlphaCare Insurance Solutions (ACIS)

The primary goal of this project is to leverage historical car insurance claim data from South Africa to optimize marketing strategies, enhance risk assessment, and identify "low-risk" client segments, ultimately leading to more competitive premium offerings and increased client attraction.

## Data Overview
The historical data covers insurance claims from February 2014 to August 2015. Key information includes:

- Policy Details:
  - UnderwrittenCoverID, PolicyID, TransactionMonth.
- Client Information:
  - IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender.
- Location Data:
  - Country, Province, PostalCode, MainCrestaZone, SubCrestaZone.
- Vehicle Details:
  - ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet.
- Plan Details:
  - SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup, Section, Product, StatutoryClass, StatutoryRiskType.
- Financials:
  - TotalPremium, TotalClaims.
## 🚀 Getting Started
To get started with the project, follow these steps:
1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/chalasimon/insurisk-analytics-and-predictive-modeling.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd insurisk-analytics-and-predictive-modeling
   ```
3. **Set Up a Python Virtual Environment** (optional but recommended):
   - If you are using `venv`:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
## 📂 Project Structure
```plaintext
insurisk-analytics-and-predictive-modeling/
├── .gitignore              # Git ignore file
├── data/
├── notebooks/              # Jupyter notebooks for 
├── scripts/                # Python scripts for data processing and analysis
├── src/                    # Source code for data processing and analysis
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation

```
### Prerequisites
- Python 3.8+

