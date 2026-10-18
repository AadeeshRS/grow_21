import React, {useState} from 'react';
import {Pressable, SafeAreaView, StyleSheet, Text, View} from 'react-native';
import {Grow21GameModule, HttpEngineClient} from '../src';

// Inlined by Expo at bundle time. Unset -> development-only MockEngine (labelled below).
const API_URL = process.env.EXPO_PUBLIC_GROW21_API_URL;
const CHILD_ID = process.env.EXPO_PUBLIC_GROW21_CHILD_ID || 'demo-child';
const engine = API_URL ? new HttpEngineClient(API_URL) : undefined;
const newSessionId = () => `session-${Date.now().toString(36)}`;

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [running, setRunning] = useState(true);
  const [status, setStatus] = useState(API_URL ? `Backend: ${API_URL}` : 'Mock engine — no learner model (set EXPO_PUBLIC_GROW21_API_URL)');
  return <SafeAreaView style={styles.root}>
    <View style={styles.banner}><Text style={styles.name}>Grow21 · Test mode · {CHILD_ID}</Text><Text style={styles.status}>{status}</Text></View>
    {running
      ? <Grow21GameModule key={sessionId} engine={engine} childId={CHILD_ID} sessionId={sessionId} locale="en-IN" mode="play" onExit={(reason) => { setRunning(false); setStatus(`Session exited: ${reason}`); }} />
      : <View style={styles.center}><Pressable accessibilityRole="button" style={styles.button} onPress={() => { setSessionId(newSessionId()); setRunning(true); }}><Text style={styles.buttonText}>Start new session</Text></Pressable></View>}
  </SafeAreaView>;
}
const styles=StyleSheet.create({root:{flex:1,backgroundColor:'#f7fbff'},banner:{padding:12,backgroundColor:'#21355a'},name:{color:'white',fontSize:16,fontWeight:'800'},status:{color:'#dceaff',fontSize:13,marginTop:3},center:{flex:1,alignItems:'center',justifyContent:'center'},button:{minHeight:64,paddingHorizontal:28,borderRadius:18,backgroundColor:'#40ad72',alignItems:'center',justifyContent:'center'},buttonText:{fontSize:18,color:'white',fontWeight:'800'}});
