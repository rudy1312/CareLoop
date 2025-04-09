import mongoose from "mongoose";

const connectDb = async () => {
    try {
        const connect = await mongoose.connect(`/`);
        console.log(`Connection Established -> ${connect.connection.host}`);        
    }catch(err) {
        console.log("Error in connection", err);
        process.exit(1);        
    }
};

export default connectDb;