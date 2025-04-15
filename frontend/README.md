# Text Classification Frontend

A modern React frontend for the text classification API, built with TypeScript and Tailwind CSS.

## Features

- Text classification
- Batch classification
- Category suggestions
- Classification validation
- Classification improvement
- Real-time feedback
- Modern UI with Tailwind CSS

## Prerequisites

- Node.js (v14 or later)
- npm or yarn
- Running backend server (http://localhost:8000)

## Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Start the development server:
```bash
npm start
# or
yarn start
```

The application will be available at http://localhost:3000.

## Project Structure

- `src/components/` - React components
  - `Home.tsx` - Home page with system status
  - `Classify.tsx` - Text classification interface
  - `Validate.tsx` - Classification validation interface
  - `Improve.tsx` - Classification improvement interface
- `src/api/` - API service functions
- `src/types/` - TypeScript type definitions
- `public/` - Static assets

## Usage

1. **Home Page**
   - View system status
   - Check model information
   - Monitor API health

2. **Classify Page**
   - Enter text to classify
   - Perform batch classification
   - Get category suggestions
   - View classification results with confidence scores

3. **Validate Page**
   - Enter text samples
   - Validate classifications
   - View accuracy scores
   - Get improvement suggestions

4. **Improve Page**
   - Enter text samples
   - Provide validation report
   - Specify categories
   - Get improved classifications

## Development

- The application uses TypeScript for type safety
- Tailwind CSS for styling
- React Router for navigation
- Axios for API requests

## Building for Production

```bash
npm run build
# or
yarn build
```

The build artifacts will be stored in the `build/` directory. 