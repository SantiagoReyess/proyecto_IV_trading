def backtesting(dataframe, theta):
    
    @dataclass
    class Operation:
        stock: str
        price: float
        shares: float

    cash = 1000000
    COM =  0.125/100

    

    active_long_positions = []
    active_short_positions = []

    portfolio_historic = [cash]

    


    data = dataframe.copy()

    for i, row in data.iterrows():

        price_1 = row.AMD
        price_2 = row.TSM

        #Update Kalman 1
        #get hedge ratio w1

        #Update Kalman 2
        #Get eigenvector for VECMS
        #get VECM

        vecm_norm = 
        


        #OPEN POSITIONS
        #OPEN (BUY A SELL B)
        if vecm_norm > theta and active_long_positions is None:
            
            available = cash * 0.40
            cost_1 = price_1 * (1 + COM)
            n_shares_long = available // cost_1

            if available > n_shares_long * cost_1:
                #BUY A
                cash -= n_shares_long * cost_1
                active_long_positions.append(
                    Operation(
                        stock = "A",
                        price = price_1,
                        shares = n_shares_long
                    )
                )

                #SELL B
                n_shares_short = hr * n_shares_long
                cost_2 = price_2 * n_shares_short * COM
                cash -= cost_2
                active_short_positions.append(
                    Operation(
                        stock = "B",
                        price = price2,
                        shares = n_shares_short
                    )
                )

    
        #OPEN (BUY B SELL A)
        if vecm_norm < theta and active_long_positions is None:
            
            available = cash * 0.40
            cost_2 = price_2 * (1 + COM)
            n_shares_long = available // cost_2

            if available > n_shares_long * cost_2:
                #BUY B
                cash -= n_shares_long * cost_2
                active_long_positions.append(
                    Operation(
                        stock = "B",
                        price = price_2,
                        shares = n_shares_long
                    )
                )

                #SELL A
                n_shares_short = n_shares_long / hr
                cost_2 = price_2 * n_shares_short * COM
                cash -= cost_2
                active_short_positions.append(
                    Operation(
                        stock = "A",
                        price = price1,
                        shares = n_shares_short
                    )
                )


        # CLOSE POSITIONS
        if abs(vecm_norm) < 0.05:

            #CLOSE LONG
            if active_long_positions[0].stock == "A":
                price_long = row.AMD
            else:
                price_long = row.TSM

            if active_short_positions[0].stock == "A":
                price_short = row.AMD
            else:
                price_short = row.TSM

            cash += price_long * active_long_positions[0].shares * (1 - COM)
            active_long_positions.clear() 

            #CLOSE SHORT
            cash += ((active_short_positions[0].price * active_short_positions[0].shares) - (price_short * active_short_positions[0].shares)) * (1 - COM) 
            active_short_positions.clear()


        # Value Portfolio for each row
        portfolio_val = 0
        portfolio_val += cash

        ## Value Long positions
        for pos in active_long_positions.copy():

            if active_long_positions[0].stock == "A":
                price_long = row.AMD
            else:
                price_long = row.TSM

            if active_short_positions[0].stock == "A":
                price_short = row.AMD
            else:
                price_short = row.TSM

            portfolio_val += price_long * pos.n_shares

        ## Value Short Positions
        for pos in active_short_positions.copy():

            if active_long_positions[0].stock == "A":
                price_long = row.AMD
            else:
                price_long = row.TSM

            if active_short_positions[0].stock == "A":
                price_short = row.AMD
            else:
                price_short = row.TSM

            portfolio_val += (pos.price * pos.n_shares) + (pos.price * pos.n_shares - price_short * pos.n_shares)

        # Add portfolio value to historic
        portfolio_historic.append(portfolio_val)

    last_close = data["Close"].iloc[-1]

    ## Close ALL Long Positions
    for pos in active_long_positions.copy():
        cash += last_close * n_shares * (1 - COM)
        active_long_positions.remove(pos)

    ## Close ALL Short Positions
    for pos in active_short_positions.copy():
        cash += (pos.price * pos.n_shares) + (pos.price - last_close) * n_shares * (1 - COM)
        active_short_positions.remove(pos)

    portfolio_val = cash
    portfolio_historic.append(portfolio_val)

    return portfolio_historic