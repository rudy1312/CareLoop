import dotenv from "dotenv";
dotenv.config();

import { app } from "./app.js";

import connectDb from "./db/dbConnection.js";

connectDb()
    .then(() => {
        app.listen(process.env.PORT, () => {
            console.log(`Server running -> ${process.env.PORT}`);            
        });
    }).catch((err) => {
        console.log("Server run err", err);        
    });