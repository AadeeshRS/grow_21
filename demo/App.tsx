import React, {useState} from 'react';
import {SafeAreaView, StyleSheet, Text, View} from 'react-native';
import {Grow21GameModule} from '../src';

export default function App() {
  const [status, setStatus] = useState('Mock session ready');
  return <SafeAreaView style={styles.root}>
    <View style={styles.banner}><Text style={styles.name}>Grow21 · Test mode</Text><Text style={styles.status}>{status}</Text></View>
    <Grow21GameModule childId="demo-child" sessionId="demo-session" locale="en-IN" mode="play" onExit={(reason) => setStatus(`Session exited: ${reason}`)} />
  </SafeAreaView>;
}
const styles=StyleSheet.create({root:{flex:1,backgroundColor:'#f7fbff'},banner:{padding:12,backgroundColor:'#21355a'},name:{color:'white',fontSize:16,fontWeight:'800'},status:{color:'#dceaff',fontSize:13,marginTop:3}});
