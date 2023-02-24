const neptune = require('@neptune.ai/neptune');
const fs = require('fs');
const axios = require('axios');

// Connect your script to Neptune new version
const myProject = 'YourUserName/YourProjectName';
const run = neptune.init({ apiToken: process.env.NEPTUNE_API_TOKEN, project: myProject });
run.stop();

const pd = require('pandas-js');
const np = require('numpy');
const { DateTime } = require('luxon');
const { createCanvas, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs-node');

// for reproducibility of our results
np.random.seed(42);
tf.random.setSeed(42);

const data_source = 'alphavantage'; // alphavantage 

if (data_source === 'alphavantage') {
    // ====================== Loading Data from Alpha Vantage ==================================
    const api_key = 'YOUR_API';
    // stock ticker symbol
    const ticker = 'AAPL'; 
    
    // JSON file with all the stock prices data 
    const url_string = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${ticker}&outputsize=full&apikey=${api_key}`;
    
    // Save data to this file
    const fileName = `stock_market_data-${ticker}.csv`;
    
    const checkExists = (filePath) => {
        try {
            fs.accessSync(filePath, fs.constants.F_OK);
            return true;
        } catch (err) {
            return false;
        }
    };
    
    const loadData = async () => {
        const csvContent = 'Date,Low,High,Close,Open\n';
        try {
            if (!checkExists(fileName)) {
                const { data } = await axios.get(url_string);
                const stockData = data['Time Series (Daily)'];
                const dataArray = [];
                for (const key in stockData) {
                    const val = stockData[key];
                    const date = DateTime.fromFormat(key, 'yyyy-MM-dd');
                    dataArray.push([date.toJSDate(), parseFloat(val['3. low']), parseFloat(val['2. high']), parseFloat(val['4. close']), parseFloat(val['1. open'])]);
                }
                dataArray.reverse();
                const contentArray = dataArray.map((row) => row.join(',')).join('\n');
                fs.writeFileSync(fileName, csvContent + contentArray);
            }
            else {
                console.log('Loading data from local');
            }
        }
        catch (err) {
            console.error(err);
        }
    };
    loadData().then(() => {
        const stockData = pd.readCsv(fileName);
        // Sort DataFrame by date
        stockData.sortValues('Date', true, false);
        
        // Define helper functions to calculate the metrics RMSE and MAPE
        const calculate_rmse = (y_true, y_pred) => {
            const rmse = np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))));
            return rmse;
        };

        // The effectiveness of prediction method is measured in terms of the Mean Absolute Percentage Error (MAPE) and RMSE
        const calculate_mape = (y_true, y_pred) => {
            const mape = np.mean(np.abs(np.divide(np.subtract(y_true, y_pred), y_true))) * 100;
            return mape;
        };
        
        // Split the time-series data into training seq X and output value Y
        const extract_seqX_outcomeY = (data, N, offset) => {
            const X = [];
            const y = [];
            
            for (let i = offset; i < data.shape[
