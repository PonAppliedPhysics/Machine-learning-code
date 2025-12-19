 # Weather Forecasting Machine Learning Project                                                   
                                                                                                   
  A machine learning project that uses Random Forest models to forecast weather conditions using real-time data from OpenWeather and NOAA APIs.
                                                                                                   
  ## 🌦️ Features                                                                                  
                                                                                                   
  - Weather data analysis and visualization                                                        
  - Machine learning forecasting with Random Forest models                                         
  - Integration with OpenWeather API for current conditions                                        
  - Integration with NOAA Climate Data API for historical data                                     
  - Secure API key management with environment variables                                           
                                                                                                   
  ## 📋 Prerequisites                                                                              
                                                                                                   
  - Python 3.x                                                                                     
  - Jupyter Notebook                                                                               
  - API Keys (free):                                                                               
    - [OpenWeather API Key](https://openweathermap.org/api) - Sign up for a free account           
    - [NOAA Climate Data API Key](https://www.ncdc.noaa.gov/cdo-web/token) - Request a free token  
                                                                                                   
  ## 🚀 Quick Start                                                                                
                                                                                                   
  ### 1. Clone the Repository                                                                      
                                                                                                   
  ```bash                                                                                          
  git clone https://github.com/PonAppliedPhysics/Machine-learning-code.git                         
  cd Machine-learning-code                                                                         
                                                                                                   
  2. Install Dependencies                                                                          
                                                                                                   
  pip install python-dotenv jupyter pandas numpy scikit-learn matplotlib requests                  
                                                                                                   
  3. Set Up API Keys                                                                               
                                                                                                   
  Create a .env file in the project root directory:                                                
                                                                                                   
  # OpenWeather API Configuration                                                                  
  OPENWEATHER_API_KEY=your_openweather_api_key_here                                                
                                                                                                   
  # NOAA Climate Data API Configuration                                                            
  NOAA_API_KEY=your_noaa_api_key_here                                                              
                                                                                                   
  Important: Replace your_openweather_api_key_here and your_noaa_api_key_here with your actual API keys.
                                                                                                   
  4. Run the Notebooks                                                                             
                                                                                                   
  jupyter notebook                                                                                 
                                                                                                   
  Open API_Keys_Guide.ipynb to test your API keys, then explore Weather_Forecasting_Machine_learning.ipynb for the main analysis.
                                                                                                   
  📁 Project Structure                                                                             
                                                                                                   
  weather-forecasting-ml/                                                                          
  ├── .env                                    # Your API keys (DO NOT COMMIT)                      
  ├── .gitignore                              # Protects sensitive files                           
  ├── API_Keys_Guide.ipynb                    # API setup and testing guide                        
  ├── Weather_Forecasting_Machine_learning.ipynb  # Main forecasting notebook                      
  ├── weather data distribution.png           # Data visualization                                 
  └── README.md                               # This file                                          
                                                                                                   
  🔐 Security                                                                                      
                                                                                                   
  - ✅ API keys are stored in .env file (not tracked by Git)                                       
  - ✅ .gitignore prevents accidental commits of sensitive data                                    
  - ✅ No hardcoded credentials in notebooks                                                       
  - ⚠️ Never commit your .env file to version control                                              
  - ⚠️ Never share your API keys publicly                                                          
                                                                                                   
  📊 What's Inside                                                                                 
                                                                                                   
  Weather_Forecasting_Machine_learning.ipynb                                                       
                                                                                                   
  - Data collection from weather APIs                                                              
  - Exploratory data analysis                                                                      
  - Feature engineering                                                                            
  - Random Forest model training                                                                   
  - Weather predictions and visualizations                                                         
                                                                                                   
  API_Keys_Guide.ipynb                                                                             
                                                                                                   
  - Step-by-step API setup instructions                                                            
  - Code examples for both APIs                                                                    
  - Testing and troubleshooting guide                                                              
                                                                                                   
  🛠️ Troubleshooting                                                                              
                                                                                                   
  API keys not loading?                                                                            
  1. Verify .env file exists in the project root                                                   
  2. Check .env format (no spaces around =)                                                        
  3. Restart Jupyter kernel                                                                        
  4. Ensure python-dotenv is installed                                                             
                                                                                                   
  API request errors?                                                                              
  - Confirm your API keys are valid and active                                                     
  - Check API rate limits (free tier restrictions)                                                 
  - Verify internet connection                                                                     
                                                                                                   
  📚 APIs Used                                                                                     
                                                                                                   
  - OpenWeather API - Current weather data and forecasts                                           
    - https://openweathermap.org/api                                                               
    - Free tier: 60 calls/minute, 1,000,000 calls/month                                            
  - NOAA Climate Data API - Historical weather observations                                        
    - https://www.ncdc.noaa.gov/cdo-web/webservices/v2                                             
    - Free tier: 5 requests/second, 10,000 requests/day                                            
                                                                                                   
  🤝 Contributing                                                                                  
                                                                                                   
  Feel free to fork this project and submit pull requests. Please ensure you don't commit any API keys or sensitive data.
                                                                                                   
  📝 License                                                                                       
                                                                                                   
  This project is open source and available for educational purposes.                              
                                                                                                   
  ⚠️ Important Notes                                                                               
                                                                                                   
  - Each user must obtain their own API keys                                                       
  - Do not share your .env file with others                                                        
  - API keys should be rotated periodically for security                                           
  - Respect API rate limits to avoid service disruption                                            
                                                                                                   
  ---                                                                                              
  🤖 Built with Claude Code                                                                        
                                                                                                   
  **Copy the entire text above** and replace your current README.md with it. This version:         
  - ✅ Does NOT include your actual API keys                                                       
  - ✅ Instructs users to get their own keys                                                       
  - ✅ Is safe to share on GitHub                                                                  
  - ✅ Provides clear setup instructions                                                           
                                                                                                   
> only for weather forecasting machine learning                                                    
                                                                                                   
⏺ # Weather Forecasting Machine Learning                                                           
                                                                                                   
  A machine learning project that uses Random Forest models to forecast weather conditions using real-time data from OpenWeather and NOAA Climate Data APIs.
                                                                                                   
  ## 🌦️ Overview                                                                                  
                                                                                                   
  This project analyzes weather patterns and builds predictive models to forecast weather conditions. It combines real-time weather data with historical climate information to train and evaluate machine learning models.
                                                                                                   
  ## 📋 Prerequisites                                                                              
                                                                                                   
  - Python 3.x                                                                                     
  - Jupyter Notebook                                                                               
  - Free API Keys:                                                                                 
    - [OpenWeather API](https://openweathermap.org/api) - Sign up for free                         
    - [NOAA Climate Data API](https://www.ncdc.noaa.gov/cdo-web/token) - Request free token        
                                                                                                   
  ## 🚀 Quick Start                                                                                
                                                                                                   
  ### 1. Clone the Repository                                                                      
                                                                                                   
  ```bash                                                                                          
  git clone https://github.com/PonAppliedPhysics/Machine-learning-code.git                         
  cd Machine-learning-code                                                                         
                                                                                                   
  2. Install Dependencies                                                                          
                                                                                                   
  pip install python-dotenv pandas numpy scikit-learn matplotlib seaborn requests jupyter          
                                                                                                   
  3. Set Up Your API Keys                                                                          
                                                                                                   
  Create a .env file in the project directory:                                                     
                                                                                                   
  OPENWEATHER_API_KEY=your_openweather_key_here                                                    
  NOAA_API_KEY=your_noaa_key_here                                                                  
                                                                                                   
  Replace with your actual API keys from:                                                          
  - OpenWeather: https://openweathermap.org/api                                                    
  - NOAA: https://www.ncdc.noaa.gov/cdo-web/token                                                  
                                                                                                   
  4. Run the Notebook                                                                              
                                                                                                   
  jupyter notebook Weather_Forecasting_Machine_learning.ipynb                                      
                                                                                                   
  The notebook will automatically load your API keys from the .env file.                           
                                                                                                   
  📁 Files                                                                                         
                                                                                                   
  - Weather_Forecasting_Machine_learning.ipynb - Main analysis and forecasting notebook            
  - API_Keys_Guide.ipynb - Setup guide and API testing                                             
  - weather data distribution.png - Data visualization                                             
  - .env - Your API keys (not tracked by Git)                                                      
  - .gitignore - Protects sensitive files                                                          
                                                                                                   
  🔐 Security                                                                                      
                                                                                                   
  ⚠️ IMPORTANT:                                                                                    
  - Never commit your .env file                                                                    
  - Never share your API keys publicly                                                             
  - Each user must use their own API keys                                                          
  - The .gitignore file protects your .env from being uploaded to GitHub                           
                                                                                                   
  📊 Features                                                                                      
                                                                                                   
  - Data Collection: Fetches real-time and historical weather data                                 
  - Data Analysis: Exploratory analysis with visualizations                                        
  - Feature Engineering: Prepares weather features for modeling                                    
  - Machine Learning: Random Forest models for weather prediction                                  
  - Evaluation: Model performance metrics and predictions                                          
                                                                                                   
  🛠️ Troubleshooting                                                                              
                                                                                                   
  API keys not loading?                                                                            
  1. Ensure .env file is in the same directory as the notebook                                     
  2. Check format: KEY_NAME=value (no spaces around =)                                             
  3. Restart Jupyter kernel                                                                        
  4. Verify python-dotenv is installed                                                             
                                                                                                   
  API errors?                                                                                      
  - Confirm API keys are valid and active                                                          
  - Check rate limits (free tiers have restrictions)                                               
  - Verify internet connection                                                                     
                                                                                                   
  📚 API Documentation                                                                             
                                                                                                   
  - https://openweathermap.org/api - Current weather & forecasts                                   
  - https://www.ncdc.noaa.gov/cdo-web/webservices/v2 - Historical climate data                     
                                                               