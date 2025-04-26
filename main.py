from bot_core import train_predict_for_ticker
if __name__ == "__main__":
    res, _ = train_predict_for_ticker("^GSPC", use_ensemble=True)
    print(res["Predicted Price for Close"], res["Trade Decision"])
