FROM node:22-bullseye

WORKDIR /app

COPY package*.json ./
RUN npm install

# Copy the rest of application file 
COPY . .

# Expose the port app runs on
EXPOSE 3000

CMD ["npm", "start"]