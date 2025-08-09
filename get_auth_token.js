#!/usr/bin/env node
/**
 * Helper script to get auth token from Supabase
 * Run this in the browser console after logging in
 */

console.log('=== Auth Token Finder ===');
console.log('Copy and paste this into your browser console after logging in to http://localhost:3000:');
console.log('');
console.log('// Method 1: Direct Supabase session');
console.log('(await window.supabase?.auth?.getSession())?.data?.session?.access_token');
console.log('');
console.log('// Method 2: Local Storage (common key patterns)');
console.log('localStorage.getItem("sb-zxebusnnyzvaktqpmuft-auth-token")');
console.log('localStorage.getItem("supabase.auth.token")');
console.log('');
console.log('// Method 3: List all localStorage items');
console.log('Object.keys(localStorage).filter(k => k.includes("auth") || k.includes("token") || k.includes("supabase")).map(k => ({key: k, value: localStorage.getItem(k)}))');
console.log('');
console.log('// Method 4: Check if logged in');
console.log('(await window.supabase?.auth?.getUser())?.data?.user');