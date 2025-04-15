import React, { useState } from 'react';
import { classifyText, classifyBatch, suggestCategories } from '../api/api';
import { ClassificationResponse, CategorySuggestionResponse } from '../types/api';

const Classify: React.FC = () => {
  const [text, setText] = useState('');
  const [categories, setCategories] = useState<string[]>([]);
  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [batchResults, setBatchResults] = useState<ClassificationResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestedCategories, setSuggestedCategories] = useState<string[]>([]);

  const handleClassify = async () => {
    if (!text) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await classifyText(text, categories.length > 0 ? categories : undefined);
      setResult(response);
    } catch (err) {
      setError('Failed to classify text');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchClassify = async () => {
    if (!text) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const texts = text.split('\n').filter(t => t.trim());
      const response = await classifyBatch(texts, categories.length > 0 ? categories : undefined);
      setBatchResults(response.results);
    } catch (err) {
      setError('Failed to classify texts');
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestCategories = async () => {
    if (!text) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const texts = text.split('\n').filter(t => t.trim());
      const response = await suggestCategories(texts);
      setSuggestedCategories(response.categories);
    } catch (err) {
      setError('Failed to suggest categories');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Text Classification</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>Enter text to classify or multiple texts (one per line) for batch classification.</p>
          </div>
          <div className="mt-5">
            <textarea
              rows={4}
              className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
              placeholder="Enter text to classify..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          <div className="mt-5">
            <div className="flex items-center space-x-4">
              <button
                type="button"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                onClick={handleClassify}
                disabled={loading}
              >
                Classify
              </button>
              <button
                type="button"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                onClick={handleBatchClassify}
                disabled={loading}
              >
                Batch Classify
              </button>
              <button
                type="button"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                onClick={handleSuggestCategories}
                disabled={loading}
              >
                Suggest Categories
              </button>
            </div>
          </div>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Classification Result</h3>
          </div>
          <div className="border-t border-gray-200">
            <dl>
              <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Category</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  {result.category}
                </dd>
              </div>
              <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Confidence</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  {(result.confidence * 100).toFixed(2)}%
                </dd>
              </div>
              <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Explanation</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  {result.explanation}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      )}

      {batchResults.length > 0 && (
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Batch Classification Results</h3>
          </div>
          <div className="border-t border-gray-200">
            <ul className="divide-y divide-gray-200">
              {batchResults.map((result, index) => (
                <li key={index} className="px-4 py-4 sm:px-6">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-primary-600 truncate">
                      Category: {result.category}
                    </p>
                    <div className="ml-2 flex-shrink-0 flex">
                      <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        {(result.confidence * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>
                  <div className="mt-2 sm:flex sm:justify-between">
                    <div className="sm:flex">
                      <p className="flex items-center text-sm text-gray-500">
                        {result.explanation}
                      </p>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {suggestedCategories.length > 0 && (
        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Suggested Categories</h3>
          </div>
          <div className="border-t border-gray-200">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex flex-wrap gap-2">
                {suggestedCategories.map((category, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium bg-primary-100 text-primary-800"
                  >
                    {category}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Classify; 