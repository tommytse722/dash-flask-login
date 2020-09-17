import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

#Email
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_html_email(receiver, subject, html_body):
    sender = "mdascalgotrading@gmail.com"
    password = "trend2020"
    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = subject
    message.attach(MIMEText(html_body, "html"))
    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(
            sender, receiver, message.as_string()
        )
    return True

def send_order_signal(receiver, tx_df, performance, execution_date):
    
    delta = 0
    last_tx_date = datetime.strftime(tx_df.index[-1] - timedelta(delta), '%Y-%m-%d')
    #last_tx_date = '2020-09-02'
    #execution_date = '2020-09-02'

    if str(last_tx_date)==str(execution_date):

        price = float(tx_df.close[-1])
        tx_shares = int(tx_df.tx_shares[-1])
        if tx_shares>0:
            action = "Buy"
        else:
            action = "Sell"
        tx_shares = abs(tx_shares)

        strategy = str(performance[0])
        stock = str(performance[1])

        no_of_win = int(performance[2])
        no_of_loss = int(performance[3])
        no_of_trade = int(performance[4])
        win_rate = no_of_win/no_of_trade
        total_win = float(performance[5])
        total_loss = float(performance[6])
        total_profit = float(performance[7])
        total_cost = float(performance[8])
        ROI = float(performance[9])

        subject = "{action} {tx_shares} shares of {stock} at ${price}"

        subject = subject.replace('{stock}', stock)
        subject = subject.replace('{action}', action)
        subject = subject.replace('{price}', str(price))
        subject = subject.replace('{tx_shares}', str(tx_shares))

        html_body = """\
          <html>
            <body>
                Win (%): {no_of_win}/{no_of_trade} ({win_rate}) <br>
                Profit (%):  {total_profit} ({ROI}) <br>
                by {strategy} on {execution_date}
            </body>
          </html>
          """
        html_body = html_body.replace('{no_of_win}', str(no_of_win))
        html_body = html_body.replace('{no_of_trade}', str(no_of_trade))
        html_body = html_body.replace('{win_rate}', '{:.1%}'.format(win_rate))
        html_body = html_body.replace('{total_profit}', str(total_profit))
        html_body = html_body.replace('{ROI}', '{:.1%}'.format(ROI))
        html_body = html_body.replace('{strategy}', strategy)
        html_body = html_body.replace('{execution_date}', str(execution_date))

        send_html_email(receiver, subject, html_body)
