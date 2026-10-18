import type {HostBridge} from './types';
export class MockHostBridge implements HostBridge { async getAvailability(){return {microphone:true,affectSignal:false}} async getAffectSignal(){return null} async startVoiceActivity(){return} async stopVoiceActivity(){return} async notifyBreak(_info:{reason:string;lengthSec:number}){return} }
